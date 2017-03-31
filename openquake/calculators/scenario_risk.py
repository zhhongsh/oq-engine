# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2017 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

import logging

import numpy

from openquake.baselib.python3compat import zip
from openquake.baselib.general import AccumDict
from openquake.commonlib import calc
from openquake.risklib import scientific
from openquake.calculators import base


F32 = numpy.float32
F64 = numpy.float64  # higher precision to avoid task order dependency


def scenario_risk(riskinput, riskmodel, monitor):
    """
    Core function for a scenario computation.

    :param riskinput:
        a of :class:`openquake.risklib.riskinput.RiskInput` object
    :param riskmodel:
        a :class:`openquake.risklib.riskinput.CompositeRiskModel` instance
    :param monitor:
        :class:`openquake.baselib.performance.Monitor` instance
    :returns:
        a dictionary {
        'agg': array of shape (E, L, R, 2),
        'avg': list of tuples (lt_idx, rlz_idx, asset_idx, statistics)
        }
        where E is the number of simulated events, L the number of loss types,
        R the number of realizations  and statistics is an array of shape
        (n, R, 4), with n the number of assets in the current riskinput object
    """
    E = monitor.oqparam.number_of_ground_motion_fields
    L = len(riskmodel.loss_types)
    R = len(riskinput.rlzs)
    I = monitor.oqparam.insured_losses + 1
    all_losses = monitor.oqparam.all_losses
    result = dict(agg=numpy.zeros((E, L * I, R), F64), avg=[],
                  all_losses=AccumDict(accum={}))
    for outputs in riskmodel.gen_outputs(riskinput, monitor):
        r = outputs.r
        assets = outputs.assets
        for l, losses in enumerate(outputs):
            if losses is None:  # this may happen
                continue
            stats = numpy.zeros((len(assets), 2, I), F32)  # mean, stddev
            for a, asset in enumerate(assets):
                stats[a, 0] = losses[a].mean()
                stats[a, 1] = losses[a].std(ddof=1)
                result['avg'].append((l, r, asset.ordinal, stats[a]))
            agglosses = losses.sum(axis=0)  # shape E, I
            for i in range(I):
                result['agg'][:, l + L * i, r] += agglosses[:, i]
            if all_losses:
                aids = [asset.ordinal for asset in outputs.assets]
                result['all_losses'][l, r] += AccumDict(zip(aids, losses))
    return result


@base.calculators.add('scenario_risk')
class ScenarioRiskCalculator(base.RiskCalculator):
    """
    Run a scenario risk calculation
    """
    core_task = scenario_risk
    pre_calculator = 'scenario'
    is_stochastic = True

    def pre_execute(self):
        """
        Compute the GMFs, build the epsilons, the riskinputs, and a dictionary
        with the unit of measure, used in the export phase.
        """
        if 'gmfs' in self.oqparam.inputs:
            self.pre_calculator = None
        base.RiskCalculator.pre_execute(self)

        logging.info('Building the epsilons')
        A = len(self.assetcol)
        E = self.oqparam.number_of_ground_motion_fields
        if self.oqparam.ignore_covs:
            eps = numpy.zeros((A, E), numpy.float32)
        else:
            eps = self.make_eps(E)
        self.datastore['etags'], gmfs = calc.get_gmfs(
            self.datastore, self.precalc)
        hazard_by_rlz = {rlz: gmfs[rlz.ordinal]
                         for rlz in self.rlzs_assoc.realizations}
        self.riskinputs = self.build_riskinputs(hazard_by_rlz, eps)

    def post_execute(self, result):
        """
        Compute stats for the aggregated distributions and save
        the results on the datastore.
        """
        loss_dt = self.oqparam.loss_dt()
        ltypes = self.riskmodel.loss_types
        I = self.oqparam.insured_losses + 1
        with self.monitor('saving outputs', autoflush=True):
            A = len(self.assetcol)

            # agg losses
            res = result['agg']
            E, LI, R = res.shape
            L = LI // I
            mean, std = scientific.mean_std(res)
            agglosses = numpy.zeros((R, LI, 2), F32)
            for l in range(LI):
                agglosses[:, l, 0] = F32(mean[l])
                agglosses[:, l, 1] = F32(std[l])

            # losses by asset
            losses_by_asset = numpy.zeros((A, R, LI, 2), F32)
            for (l, r, aid, stat) in result['avg']:
                for i in range(I):
                    losses_by_asset[aid, r, l + L * i] = stat[:, i]
            self.datastore['losses_by_asset'] = losses_by_asset
            self.datastore['agglosses-rlzs'] = agglosses

            if self.oqparam.all_losses:
                array = numpy.zeros((A, E, R), loss_dt)
                for (l, r), losses_by_aid in result['all_losses'].items():
                    for aid in losses_by_aid:
                        lba = losses_by_aid[aid]  # (E, I)
                        array[ltypes[l]][aid, :, r] = lba[:, 0]
                        if I == 2:
                            array[ltypes[l] + '_ins'][aid, :, r] = lba[:, 1]
                self.datastore['all_losses-rlzs'] = array
