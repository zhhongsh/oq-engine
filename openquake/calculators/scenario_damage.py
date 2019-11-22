# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2019 GEM Foundation
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

import h5py
import numpy
from openquake.baselib import hdf5, general
from openquake.risklib import scientific
from openquake.calculators import base

U8 = numpy.uint8
U16 = numpy.uint16
U32 = numpy.uint32
F32 = numpy.float32
F64 = numpy.float64

edn_dt = numpy.dtype([('eid', U32), ('dsi', U8), ('n', U16)])


def discrete_damage_state_distribution(fractions, eids, number):
    """
    :param fractions: an array of probabilities of shape (F, D)
    :param number: an integer in the range 0 .. 65535
    :returns: the damage state distribution for each event in terms of integers

    >>> fractions = numpy.array([[.8, .1, .1], [.7, .2, .1]])  # shape (2, 3)
    >>> discrete_damage_state_distribution(fractions, [0, 1], 100)
    [(0, array([10, 10], dtype=uint16)), (1, array([20, 10], dtype=uint16))]
    """
    ddd = []
    for eid, fracs in zip(eids, fractions):
        n = U16(numpy.round(fracs[1:] * number))
        if n.any():
            ddd.append((eid, n))
    return ddd


def scenario_damage(riskinputs, crmodel, param, monitor):
    """
    Core function for a damage computation.

    :param riskinputs:
        :class:`openquake.risklib.riskinput.RiskInput` objects
    :param crmodel:
        a :class:`openquake.risklib.riskinput.CompositeRiskModel` instance
    :param monitor:
        :class:`openquake.baselib.performance.Monitor` instance
    :param param:
        dictionary of extra parameters
    :returns:
        a dictionary {'d_asset': [(l, r, a, mean-stddev), ...],
                      'd_event': damage array of shape R, L, F, D,
                      'c_asset': [(l, r, a, mean-stddev), ...],
                      'c_event': damage array of shape R, L, F}

    `d_asset` and `d_tag` are related to the damage distributions
    whereas `c_asset` and `c_tag` are the consequence distributions.
    If there is no consequence model `c_asset` is an empty list and
    `c_tag` is a zero-valued array.
    """
    L = len(crmodel.loss_types)
    D = len(crmodel.damage_states)
    F = param['number_of_ground_motion_fields']
    R = riskinputs[0].hazard_getter.num_rlzs
    result = dict(d_asset=[], d_event=numpy.zeros((F, R, L, D), F64),
                  c_asset=[], c_event=numpy.zeros((F, R, L), F64))
    if param['asset_damage_table']:
        adl = general.AccumDict(accum=[])  # a, l -> ddd
    for ri in riskinputs:
        for out in ri.gen_outputs(crmodel, monitor):
            r = out.rlzi
            for l, loss_type in enumerate(crmodel.loss_types):
                for asset, fractions in zip(ri.assets, out[loss_type]):
                    dmg = fractions[:, :D] * asset['number']  # shape (F, D)
                    if param['asset_damage_table']:
                        eids = numpy.arange(r*F, r*F + F, dtype=U32)
                        ddd = discrete_damage_state_distribution(
                            fractions[:, :D], eids, asset['number'])
                        adl[asset['ordinal'], l].extend(ddd)
                    result['d_event'][:, r, l] += dmg
                    result['d_asset'].append(
                        (l, r, asset['ordinal'], scientific.mean_std(dmg)))
                    if crmodel.has('consequence'):
                        csq = fractions[:, D] * asset['value-' + loss_type]
                        result['c_asset'].append(
                            (l, r, asset['ordinal'], scientific.mean_std(csq)))
                        result['c_event'][:, r, l] += csq
    if param['asset_damage_table']:
        result['asset_damage_table'] = adl
    return result


@base.calculators.add('scenario_damage')
class ScenarioDamageCalculator(base.RiskCalculator):
    """
    Scenario damage calculator
    """
    core_task = scenario_damage
    is_stochastic = True
    precalc = 'scenario'
    accept_precalc = ['scenario']

    def pre_execute(self):
        super().pre_execute()
        F = self.oqparam.number_of_ground_motion_fields
        self.param['number_of_ground_motion_fields'] = F
        self.param['tags'] = list(self.assetcol.tagcol)
        self.param['asset_damage_table'] = self.oqparam.asset_damage_table
        self.riskinputs = self.build_riskinputs('gmf')

    def post_execute(self, result):
        """
        Compute stats for the aggregated distributions and save
        the results on the datastore.
        """
        if not result:
            self.collapsed()
            return
        dstates = self.crmodel.damage_states
        ltypes = self.crmodel.loss_types
        A = len(self.assetcol)
        L = len(ltypes)
        R = len(self.rlzs_assoc.realizations)
        D = len(dstates)
        N = len(self.assetcol)
        F = self.oqparam.number_of_ground_motion_fields

        # damage distributions
        dt_list = []
        mean_std_dt = numpy.dtype([('mean', (F32, D)), ('stddev', (F32, D))])
        for ltype in ltypes:
            dt_list.append((ltype, mean_std_dt))
        d_asset = numpy.zeros((N, R, L, 2, D), F32)
        for (l, r, a, stat) in result['d_asset']:
            d_asset[a, r, l] = stat
        self.datastore['dmg_by_asset'] = d_asset
        dmg_dt = [(ds, F32) for ds in self.crmodel.damage_states]
        d_event = numpy.zeros((F, R, L), dmg_dt)
        for d, ds in enumerate(self.crmodel.damage_states):
            d_event[ds] = result['d_event'][:, :, :, d]
        self.datastore['dmg_by_event'] = d_event

        # asset_damage_table
        if self.oqparam.asset_damage_table:
            dt = numpy.dtype([('eid', numpy.uint32), ('ddd', (U16, D - 1))])
            vddd = h5py.special_dtype(vlen=dt)
            adt = self.datastore.create_dset(
                'asset_damage_table', vddd, (A, L), fillvalue=None)
            eff_events = []
            for al, ddd in result['asset_damage_table'].items():
                adt[al] = ddd
                eff_events.append(len(ddd))
            adt.attrs['eff_events'] = numpy.average(eff_events)
            adt.attrs['number'] = U16(self.assetcol['number'])

        # consequence distributions
        if result['c_asset']:
            dtlist = [('event_id', U32), ('rlz_id', U16),
                      ('loss', (F32, (L,)))]
            stat_dt = numpy.dtype([('mean', F32), ('stddev', F32)])
            c_asset = numpy.zeros((N, R, L), stat_dt)
            for (l, r, a, stat) in result['c_asset']:
                c_asset[a, r, l] = stat
            self.datastore['losses_by_asset'] = c_asset
            self.datastore['losses_by_event'] = numpy.fromiter(
                ((eid + rlzi * F, rlzi, F32(result['c_event'][eid, rlzi]))
                 for rlzi in range(R) for eid in range(F)), dtlist)
