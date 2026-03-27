import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List
from copy import deepcopy
import os


# ПАРАМЕТРЫ

@dataclass
class SGParams:
    """Параметры одного синхронного генератора (дизель-генератор)."""
    name: str = "СГ"
    S_nom: float = 400.0
    P_nom: float = 320.0
    V_nom: float = 1.0
    H: float = 3.5
    D_mech: float = 2.0

    Xd: float = 1.2
    Xq: float = 0.8
    Xd_prime: float = 0.3
    Xd_dprime: float = 0.18
    Td0_prime: float = 5.0
    Td0_dprime: float = 0.035
    Ra: float = 0.01

    R_droop: float = 0.05
    T_act: float = 0.1
    T_gov: float = 0.5
    ramp_up: float = 0.08
    ramp_down: float = 0.15
    P_min: float = 0.05
    P_max: float = 1.0

    Ka: float = 100.0
    Ta: float = 0.01
    Tb: float = 10.0
    Tc: float = 1.0
    Ke: float = 1.0
    Te: float = 0.5
    Vref: float = 1.0
    Efd_min: float = 0.0
    Efd_max: float = 5.0
    Xc: float = 0.05

    X_line_sg: float = 0.0
    R_line_sg: float = 0.0


@dataclass
class VSGParams:
    """Параметры ВСГ (инвертор + BESS)."""
    name: str = "ВСГ"
    S_nom: float = 300.0
    P_nom: float = 250.0

    H_virt: float = 1.5
    D_virt: float = 5.0

    R_droop: float = 0.03
    K_rocof: float = 0.0
    T_rocof: float = 0.02
    K_ff: float = 0.0

    T_inv: float = 0.005
    zeta_inv: float = 0.7
    I_max: float = 1.2
    P_max: float = 1.0
    P_min: float = -0.5

    Kp_v: float = 10.0
    Ki_v: float = 50.0
    T_qv: float = 0.005
    Vref: float = 1.0

    Xv: float = 0.10

    E_batt_kWh: float = 200.0
    SoC_init: float = 0.80
    SoC_min: float = 0.10
    SoC_max: float = 0.95
    eta_charge: float = 0.95
    eta_discharge: float = 0.95

    T_restore: float = 0.0
    T_washout: float = 0.0


@dataclass
class BusParams:
    """Параметры шины."""
    f_nom: float = 50.0
    V_nom: float = 1.0
    X_line: float = 0.05
    R_line: float = 0.01
    D_load: float = 1.0


@dataclass
class LoadProfile:
    events: List[tuple] = field(default_factory=lambda: [
        (0.0, 400.0, 200.0),
        (5.0, 700.0, 350.0),
    ])


@dataclass
class SimParams:
    dt: float = 0.001
    T_end: float = 60.0
    downsample: int = 10



# МОДЕЛЬ СГ

class SynchronousGenerator:
    """
    Синхронный генератор - модель 4-го порядка.
    Состояния: delta, omega, Eq_prime, Eq_dprime, Efd, P_valve, P_mech.
    """

    def __init__(self, params: SGParams, idx: int = 0):
        self.p = params
        self.idx = idx
        self.w0 = 2 * np.pi * 50.0

        self.delta = 0.0
        self.omega = self.w0
        self.Eq_prime = 1.0
        self.Eq_dprime = 1.0
        self.Efd = 1.0
        self.Efd0 = 1.0
        self.Vr = 0.0
        self.P_valve = 0.0
        self.P_mech = 0.0
        self.P_set = 0.0

        self.Pe = 0.0
        self.Qe = 0.0
        self.Vt = 1.0
        self.Id = 0.0
        self.Iq = 0.0

    def init_steady_state(self, P0_pu, Q0_pu, V_bus):
        p = self.p
        Vt = V_bus
        self.Vt = Vt
        self.Pe = P0_pu
        self.Qe = Q0_pu

        X_total = p.Xd_dprime + p.X_line_sg

        denom = Q0_pu + Vt ** 2 / X_total
        if abs(denom) > 1e-9:
            self.delta = np.arctan2(P0_pu, denom)
        else:
            self.delta = 0.0

        sin_d = np.sin(self.delta)
        cos_d = np.cos(self.delta)

        if abs(sin_d) > 1e-9:
            self.Eq_dprime = P0_pu * X_total / (Vt * sin_d)
        else:
            self.Eq_dprime = Vt + Q0_pu * X_total / (Vt + 1e-9)

        self.Id = (self.Eq_dprime - Vt * cos_d) / X_total
        self.Iq = Vt * sin_d / X_total

        self.Eq_prime = self.Eq_dprime + (p.Xd_prime - p.Xd_dprime) * self.Id

        self.Efd = self.Eq_prime + (p.Xd - p.Xd_prime) * self.Id
        self.Efd = np.clip(self.Efd, p.Efd_min, p.Efd_max)
        self.Efd0 = self.Efd

        self.P_mech = P0_pu
        self.P_valve = P0_pu
        self.P_set = P0_pu
        self.omega = self.w0

    def step(self, dt, V_bus, P_set_external, f_bus=None):
        p = self.p
        w0 = self.w0

        X_total = p.Xd_dprime + p.X_line_sg
        R_total = p.Ra + p.R_line_sg

        if V_bus > 0.01:
            sin_d = np.sin(self.delta)
            cos_d = np.cos(self.delta)

            self.Pe = (self.Eq_dprime * V_bus / X_total) * sin_d
            self.Qe = (self.Eq_dprime * V_bus * cos_d - V_bus ** 2) / X_total

            self.Id = (self.Eq_dprime - V_bus * cos_d) / X_total
            self.Iq = V_bus * sin_d / X_total

            Vd_t = V_bus * cos_d + R_total * self.Id - p.X_line_sg * self.Iq
            Vq_t = V_bus * sin_d + R_total * self.Iq + p.X_line_sg * self.Id
            self.Vt = np.sqrt(Vd_t ** 2 + Vq_t ** 2)
        else:
            self.Pe = self.Qe = self.Id = self.Iq = 0.0
            self.Vt = 0.0

        if f_bus is not None:
            dw_gov = (f_bus - 50.0) / 50.0
        else:
            dw_gov = (self.omega - w0) / w0

        P_gov_ref = P_set_external - dw_gov / p.R_droop
        P_gov_ref = np.clip(P_gov_ref, p.P_min, p.P_max)

        dP_valve_dt = (P_gov_ref - self.P_valve) / p.T_act
        dP_valve_dt = np.clip(dP_valve_dt, -p.ramp_down, p.ramp_up)
        self.P_valve += dP_valve_dt * dt
        self.P_valve = np.clip(self.P_valve, p.P_min, p.P_max)

        dP_mech_dt = (self.P_valve - self.P_mech) / p.T_gov
        self.P_mech += dP_mech_dt * dt
        self.P_mech = np.clip(self.P_mech, p.P_min, p.P_max)

        dw_pu = (self.omega - w0) / w0
        P_accel = self.P_mech - self.Pe - p.D_mech * dw_pu
        d_omega = (P_accel / (2.0 * p.H)) * w0 * dt
        self.omega += d_omega
        self.delta += (self.omega - w0) * dt
        self.delta = (self.delta + np.pi) % (2 * np.pi) - np.pi

        V_err = p.Vref - self.Vt
        Efd_target = self.Efd0 + p.Ka * V_err
        Efd_target = np.clip(Efd_target, p.Efd_min, p.Efd_max)

        dEfd = (Efd_target - self.Efd) / p.Te * dt
        self.Efd += dEfd
        self.Efd = np.clip(self.Efd, p.Efd_min, p.Efd_max)

        dEq_p = (self.Efd - self.Eq_prime
                 - (p.Xd - p.Xd_prime) * self.Id) / p.Td0_prime
        self.Eq_prime += dEq_p * dt

        dEq_dp = (self.Eq_prime - self.Eq_dprime
                  - (p.Xd_prime - p.Xd_dprime) * self.Id) / p.Td0_dprime
        self.Eq_dprime += dEq_dp * dt

        return self.Pe * self.p.S_nom, self.Qe * self.p.S_nom


# МОДЕЛЬ ВСГ (с ограничением виртуальной частоты)

class VirtualSynchronousGenerator:
    """
    ВСГ - grid-forming инвертор с виртуальным swing equation.
    Модификация: ограничение виртуальной частоты +-2 Гц от номинала.
    """

    def __init__(self, params: VSGParams):
        self.p = params
        self.w0 = 2 * np.pi * 50.0

        self.omega_virt = self.w0
        self.delta_virt = 0.0
        self.P_ref = 0.0
        self.P_out = 0.0
        self.Q_ref = 0.0
        self.Q_out = 0.0
        self.SoC = params.SoC_init
        self.V_int_err = 0.0
        self.P_set = 0.0
        self.f_prev = 50.0
        self.dfdt_filt = 0.0
        self.P_deficit_ff = 0.0
        self.dP_out = 0.0
        self.P_restore = 0.0
        self.P_ff_filt = 0.0

    def init_steady_state(self, P0_pu, Q0_pu, V_bus):
        self.P_out = P0_pu
        self.P_ref = P0_pu
        self.P_set = P0_pu
        self.Q_out = Q0_pu
        self.Q_ref = Q0_pu
        self.omega_virt = self.w0
        self.delta_virt = 0.0
        self.V_int_err = Q0_pu / (self.p.Ki_v + 1e-9)

    def step(self, dt, V_bus, f_bus):
        p = self.p
        w0 = self.w0

        f_bus_hz = f_bus
        df_bus_pu = (f_bus_hz - 50.0) / 50.0
        P_droop = -df_bus_pu / p.R_droop

        if dt > 1e-9:
            dfdt_raw = (f_bus_hz - self.f_prev) / dt
        else:
            dfdt_raw = 0.0
        self.f_prev = f_bus_hz

        alpha_rocof = dt / (p.T_rocof + dt)
        self.dfdt_filt += alpha_rocof * (dfdt_raw - self.dfdt_filt)

        dfdt_pu = self.dfdt_filt / 50.0
        P_rocof = -p.K_rocof * dfdt_pu

        P_deficit_raw = p.K_ff * self.P_deficit_ff

        if p.T_washout > 0.01:
            alpha_wo = dt / (p.T_washout + dt)
            self.P_ff_filt += alpha_wo * (P_deficit_raw - self.P_ff_filt)
            P_ff = P_deficit_raw - self.P_ff_filt
        else:
            P_ff = P_deficit_raw

        dw_pu = (self.omega_virt - w0) / w0
        P_total_ref = self.P_set + P_droop + P_rocof + P_ff
        P_accel = P_total_ref - self.P_out - p.D_virt * dw_pu

        if p.H_virt > 0.01:
            d_omega = (P_accel / (2.0 * p.H_virt)) * w0 * dt
        else:
            d_omega = 0.0

        self.omega_virt += d_omega
        self.delta_virt += (self.omega_virt - w0) * dt

        # >>> МОДИФИКАЦИЯ: ограничение виртуальной частоты +-2 Гц
        w_min = 2 * np.pi * 48.0
        w_max = 2 * np.pi * 52.0
        self.omega_virt = np.clip(self.omega_virt, w_min, w_max)

        if p.T_restore > 0.01:
            self.P_restore += (self.P_out - self.P_set) / p.T_restore * dt
            self.P_restore = np.clip(self.P_restore, -1.0, 1.0)
        else:
            self.P_restore = 0.0

        self.P_ref = P_total_ref - self.P_restore
        self.P_ref = np.clip(self.P_ref, p.P_min, p.P_max)

        if self.SoC <= p.SoC_min and self.P_ref > 0:
            self.P_ref = 0.0
        elif self.SoC >= p.SoC_max and self.P_ref < 0:
            self.P_ref = 0.0

        omega_n = 2.0 * np.pi / (4.0 * p.T_inv)
        zeta = p.zeta_inv

        ddP = omega_n ** 2 * (self.P_ref - self.P_out) - 2.0 * zeta * omega_n * self.dP_out
        self.dP_out += ddP * dt
        self.P_out += self.dP_out * dt
        self.P_out = np.clip(self.P_out, p.P_min, p.P_max)

        V_error = p.Vref - V_bus

        self.V_int_err += V_error * dt
        Q_max_avail = np.sqrt(max(0, p.I_max ** 2 - self.P_out ** 2))
        if (self.Q_out >= Q_max_avail and V_error > 0) or \
                (self.Q_out <= -Q_max_avail and V_error < 0):
            self.V_int_err -= V_error * dt
        self.V_int_err = np.clip(self.V_int_err, -1.0, 1.0)

        self.Q_ref = p.Kp_v * V_error + p.Ki_v * self.V_int_err

        dQ = (self.Q_ref - self.Q_out) / p.T_qv * dt
        self.Q_out += dQ

        S_out = np.sqrt(self.P_out ** 2 + self.Q_out ** 2)
        if S_out > p.I_max:
            Q_max = np.sqrt(max(0, p.I_max ** 2 - self.P_out ** 2))
            self.Q_out = np.clip(self.Q_out, -Q_max, Q_max)

        P_batt_kW = self.P_out * p.P_nom
        if P_batt_kW >= 0:
            energy_kWh = P_batt_kW / p.eta_discharge * (dt / 3600.0)
        else:
            energy_kWh = P_batt_kW * p.eta_charge * (dt / 3600.0)

        self.SoC -= energy_kWh / p.E_batt_kWh
        self.SoC = np.clip(self.SoC, 0.0, 1.0)

        P_kW = self.P_out * p.P_nom
        Q_kvar = self.Q_out * p.S_nom
        return P_kW, Q_kvar


# МОДЕЛЬ ШИНЫ

class BusModel:
    """
    Модель общей шины - Center of Inertia для частоты,
    Q-V чувствительность для напряжения.
    Быстрое dP_vsg/dt -> кратковременный dV на шине.
    """

    def __init__(self, params: BusParams):
        self.p = params
        self.f = params.f_nom
        self.V = params.V_nom
        self.w0 = 2 * np.pi * params.f_nom
        self.P_vsg_prev = 0.0

    def step(self, dt, P_gen_total_kW, Q_gen_total_kvar,
             P_load_base_kW, Q_load_base_kvar,
             H_total, S_total_kVA,
             omega_coi,
             P_vsg_kW=0.0, Q_vsg_kvar=0.0, Xv_vsg=0.1):
        p = self.p
        f_nom = p.f_nom
        w0 = self.w0

        df_pu = (self.f - f_nom) / f_nom
        P_load_kW = P_load_base_kW * (1.0 + p.D_load * df_pu)
        Q_load_kvar = Q_load_base_kvar

        P_imbalance_pu = (P_gen_total_kW - P_load_kW) / (S_total_kVA + 1e-9)
        D_sys = 2.0
        df_dt = (P_imbalance_pu - D_sys * df_pu) / (2.0 * max(H_total, 0.5)) * f_nom
        self.f += df_dt * dt
        self.f = np.clip(self.f, 47.0, 53.0)

        dP_pu = (P_gen_total_kW - P_load_kW) / (S_total_kVA + 1e-9)
        Q_imbalance_pu = (Q_gen_total_kvar - Q_load_kvar) / (S_total_kVA + 1e-9)

        if dt > 1e-9:
            dPdt_vsg = (P_vsg_kW - self.P_vsg_prev) / dt
        else:
            dPdt_vsg = 0.0
        self.P_vsg_prev = P_vsg_kW
        dPdt_vsg_pu = dPdt_vsg / (S_total_kVA + 1e-9)
        T_emf = 0.02
        V_emf_transient = Xv_vsg * dPdt_vsg_pu * T_emf

        V_target = (p.V_nom
                    + p.R_line * dP_pu
                    + p.X_line * Q_imbalance_pu
                    + V_emf_transient)

        T_v = 0.02
        self.V += (V_target - self.V) / T_v * dt
        self.V = np.clip(self.V, 0.85, 1.15)

        return self.f, self.V, P_load_kW, Q_load_kvar


# СИМУЛЯЦИЯ

def get_load_at_time(t, load_profile: LoadProfile):
    P, Q = load_profile.events[0][1], load_profile.events[0][2]
    for ev_t, ev_P, ev_Q in load_profile.events:
        if t >= ev_t:
            P, Q = ev_P, ev_Q
        else:
            break
    return P, Q


def run_simulation(
        n_sg: int = 3,
        sg_params=None,
        vsg_params: VSGParams = None,
        bus_params: BusParams = None,
        load_profile: LoadProfile = None,
        sim_params: SimParams = None,
        P0_sg_pu: list = None,
        P_set_schedule: list = None,
):
    if sg_params is None: sg_params = SGParams()
    if vsg_params is None: vsg_params = VSGParams()
    if bus_params is None: bus_params = BusParams()
    if load_profile is None: load_profile = LoadProfile()
    if sim_params is None: sim_params = SimParams()

    if isinstance(sg_params, list):
        sg_params_list = sg_params
        assert len(sg_params_list) == n_sg
    else:
        sg_params_list = [sg_params] * n_sg

    dt = sim_params.dt
    N = int(sim_params.T_end / dt)
    ds = sim_params.downsample
    w0 = 2 * np.pi * bus_params.f_nom

    sg_list = [SynchronousGenerator(sg_params_list[i], idx=i) for i in range(n_sg)]
    vsg = VirtualSynchronousGenerator(vsg_params)
    bus = BusModel(bus_params)

    P0_load, Q0_load = get_load_at_time(0, load_profile)
    S_sg_total = sum(sp.S_nom for sp in sg_params_list)
    S_total = S_sg_total + vsg_params.S_nom

    if P0_sg_pu is not None:
        assert len(P0_sg_pu) == n_sg
        for i, sg in enumerate(sg_list):
            sp = sg_params_list[i]
            sg_P0 = P0_sg_pu[i]
            sg_Q0 = Q0_load * (sp.S_nom / S_total) / sp.S_nom
            sg.init_steady_state(sg_P0, sg_Q0, bus_params.V_nom)
    else:
        for i, sg in enumerate(sg_list):
            sp = sg_params_list[i]
            sg_P0 = (P0_load * sp.S_nom / S_sg_total) / sp.S_nom
            sg_Q0 = Q0_load * (sp.S_nom / S_total) / sp.S_nom
            sg.init_steady_state(sg_P0, sg_Q0, bus_params.V_nom)

    vsg_P0 = 0.0
    vsg_Q0 = Q0_load * (vsg_params.S_nom / S_total) / vsg_params.S_nom
    vsg.init_steady_state(vsg_P0, vsg_Q0, bus_params.V_nom)

    H_total = (sum(sp.H * sp.S_nom for sp in sg_params_list) +
               vsg_params.H_virt * vsg_params.S_nom) / S_total

    P_set_fixed = []
    for i in range(n_sg):
        if P0_sg_pu is not None:
            P_set_fixed.append(P0_sg_pu[i])
        else:
            sp = sg_params_list[i]
            P_set_fixed.append((P0_load * sp.S_nom / S_sg_total) / sp.S_nom)

    n_out = N // ds + 2
    out = {
        't': np.zeros(n_out),
        'f': np.zeros(n_out),
        'V': np.zeros(n_out),
        'P_sg_total': np.zeros(n_out),
        'Q_sg_total': np.zeros(n_out),
        'P_sg_each': np.zeros((n_sg, n_out)),
        'Q_sg_each': np.zeros((n_sg, n_out)),
        'P_mech': np.zeros((n_sg, n_out)),
        'Efd': np.zeros((n_sg, n_out)),
        'Vt_sg': np.zeros((n_sg, n_out)),
        'P_vsg': np.zeros(n_out),
        'Q_vsg': np.zeros(n_out),
        'P_load': np.zeros(n_out),
        'Q_load': np.zeros(n_out),
        'SoC': np.zeros(n_out),
        'omega_vsg': np.zeros(n_out),
    }
    oi = 0

    for k in range(N):
        t = k * dt
        V_bus = bus.V
        f_bus = bus.f

        P_load_base, Q_load_base = get_load_at_time(t, load_profile)

        P_sg_sum_kW = 0.0
        Q_sg_sum_kvar = 0.0

        event_happened = False
        if P_set_schedule is not None:
            for t_ev, _ in P_set_schedule:
                if t >= t_ev:
                    event_happened = True
                    break

        for i, sg in enumerate(sg_list):
            sp = sg_params_list[i]
            P_set = P_set_fixed[i]
            if P_set_schedule is not None:
                for t_ev, p_set_list in P_set_schedule:
                    if t >= t_ev:
                        P_set = p_set_list[i]
            if P_set < 0.001:
                sg.Pe = 0.0
                sg.Qe = 0.0
                sg.P_mech = 0.0
                sg.P_valve = 0.0
                sg.omega = sg.w0
                Pe_kW = 0.0
                Qe_kvar = 0.0
            elif not event_happened:
                P_set = np.clip(P_set, sp.P_min, sp.P_max)
                sg.Pe = P_set
                sg.P_mech = P_set
                sg.P_valve = P_set
                Pe_kW = P_set * sp.S_nom
                Qe_kvar = sg.Qe * sp.S_nom
            else:
                P_set = np.clip(P_set, sp.P_min, sp.P_max)
                Pe_kW, Qe_kvar = sg.step(dt, V_bus, P_set, f_bus=f_bus)

            P_sg_sum_kW += Pe_kW
            Q_sg_sum_kvar += Qe_kvar

        vsg.P_set = 0.0
        P_deficit_kW = P_load_base - P_sg_sum_kW
        vsg.P_deficit_ff = max(0.0, P_deficit_kW) / vsg_params.P_nom

        P_vsg_kW, Q_vsg_kvar = vsg.step(dt, V_bus, f_bus)

        P_gen_total = P_sg_sum_kW + P_vsg_kW
        Q_gen_total = Q_sg_sum_kvar + Q_vsg_kvar

        H_S_sum = 0.0
        H_S_omega_sum = 0.0
        for i_sg, sg_unit in enumerate(sg_list):
            sp = sg_params_list[i_sg]
            if sg_unit.Pe < 1e-6 and sg_unit.P_mech < 1e-6:
                continue
            HS = sp.H * sp.S_nom
            H_S_sum += HS
            H_S_omega_sum += HS * sg_unit.omega
        HS_vsg = vsg_params.H_virt * vsg_params.S_nom
        H_S_sum += HS_vsg
        H_S_omega_sum += HS_vsg * vsg.omega_virt
        omega_coi = H_S_omega_sum / (H_S_sum + 1e-9)

        f_bus, V_bus, P_load_actual, Q_load_actual = bus.step(
            dt, P_gen_total, Q_gen_total,
            P_load_base, Q_load_base,
            H_total, S_total,
            omega_coi,
            P_vsg_kW=P_vsg_kW, Q_vsg_kvar=Q_vsg_kvar,
            Xv_vsg=vsg_params.Xv
        )

        if k % ds == 0 and oi < n_out:
            out['t'][oi] = t
            out['f'][oi] = f_bus
            out['V'][oi] = V_bus
            out['P_sg_total'][oi] = P_sg_sum_kW
            out['Q_sg_total'][oi] = Q_sg_sum_kvar
            out['P_vsg'][oi] = P_vsg_kW
            out['Q_vsg'][oi] = Q_vsg_kvar
            out['P_load'][oi] = P_load_actual
            out['Q_load'][oi] = Q_load_actual
            out['SoC'][oi] = vsg.SoC
            out['omega_vsg'][oi] = vsg.omega_virt
            for i, sg in enumerate(sg_list):
                out['P_sg_each'][i, oi] = sg.Pe * sg.p.S_nom
                out['Q_sg_each'][i, oi] = sg.Qe * sg.p.S_nom
                out['P_mech'][i, oi] = sg.P_mech * sg.p.S_nom
                out['Efd'][i, oi] = sg.Efd
                out['Vt_sg'][i, oi] = sg.Vt
            oi += 1

    for key in out:
        if isinstance(out[key], np.ndarray):
            if out[key].ndim == 1:
                out[key] = out[key][:oi]
            else:
                out[key] = out[key][:, :oi]

    return out


# ГРАФИКИ

def _setup_sci_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 8.5,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 1.4,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.4,
        'grid.linestyle': '--',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '0.85',
        'legend.fancybox': False,
    })


_C = {
    'sg': '#2166AC', 'vsg': '#1B7837', 'load': '#D6604D',
    'gen': '#878787', 'freq': '#7B3294', 'volt': '#E08214',
    'efd': '#B2182B', 'imb': '#C51B7D', 'fvsg': '#66C2A5',
    'qsg': '#4393C3', 'qvsg': '#66C2A5', 'qload': '#FC8D62',
}

_SG_COLORS = ['#2166AC', '#E7298A', '#D95F02']
_EVENT_TIMES = [5.0]


def _draw_event_lines(ax, times=None, color='0.5', lw=0.8, ls='--', alpha=0.5):
    for te in (times or _EVENT_TIMES):
        ax.axvline(te, color=color, lw=lw, ls=ls, alpha=alpha)


def _single_fig(w=10, h=4.5):
    """Создать одиночную фигуру."""
    _setup_sci_style()
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    return fig, ax


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  Сохранено: {path}")
    plt.close(fig)


def plot_showcase_a(res, n_sg, sg_params_list, vsg_p, save_path=None):
    """Активная мощность каждой машины."""
    fig, ax = _single_fig()
    t = res['t']
    for i in range(n_sg):
        ax.plot(t, res['P_sg_each'][i] / 1000, color=_SG_COLORS[i],
                lw=1.5, label=f'СГ-{i+1}')
    ax.plot(t, res['P_vsg'] / 1000, color=_C['vsg'], lw=2.0,
            label=f'ВСГ (H_virt={vsg_p.H_virt}с)', zorder=10)
    _draw_event_lines(ax)
    ax.axhline(0, color='0.7', lw=0.4, ls=':')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('P, МВт')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_title('Активная мощность каждой машины - противофазные колебания',
                 loc='left', fontsize=10)
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    _save(fig, save_path)


def plot_showcase_b(res, vsg_p, save_path=None):
    """Частота шины и виртуальная частота ВСГ."""
    fig, ax = _single_fig()
    t = res['t']
    f_vsg = res['omega_vsg'] / (2 * np.pi)
    ax.plot(t, res['f'], color=_C['freq'], lw=1.5, label='f шины')
    ax.plot(t, f_vsg, color=_C['vsg'], lw=1.2, ls='--', label='f ВСГ (виртуальная)')
    ax.axhline(50.0, color='0.7', lw=0.4, ls=':')
    _draw_event_lines(ax)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('f, Гц')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Частота шины - ВСГ навязывает колебания', loc='left', fontsize=10)
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    _save(fig, save_path)


def plot_showcase_c(res, n_sg, save_path=None):
    """Отклонения dP от среднего - противофаза."""
    fig, ax = _single_fig()
    t = res['t']
    P_sg_mean = np.mean([res['P_sg_each'][i] for i in range(n_sg)], axis=0)
    for i in range(n_sg):
        dP = (res['P_sg_each'][i] - P_sg_mean) / 1000
        ax.plot(t, dP, color=_SG_COLORS[i], lw=1.5,
                label=f'dP СГ-{i+1} (от среднего)')
    ax.axhline(0, color='0.7', lw=0.5, ls=':')
    _draw_event_lines(ax)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('ΔP, МВт')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Отклонения P от среднего - видна противофаза колебаний',
                 loc='left', fontsize=10)
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    _save(fig, save_path)


def plot_showcase_d(res, save_path=None):
    """Напряжение шины."""
    fig, ax = _single_fig()
    t = res['t']
    ax.plot(t, res['V'], color=_C['volt'], lw=1.5, label='V шины')
    ax.axhline(1.0, color='0.7', lw=0.4, ls=':')
    _draw_event_lines(ax)
    ax.fill_between(t, 0.95, 1.05, alpha=0.04, color='green')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('V, о.е.')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Напряжение шины - колебания от перетоков P и Q',
                 loc='left', fontsize=10)
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    _save(fig, save_path)


def _zoom_mask(res, event_time, window):
    t = res['t']
    t0, t1 = event_time - 1.0, event_time + window
    mask = (t >= t0) & (t <= t1)
    return t[mask], mask


def plot_zoom_a(res, n_sg, event_time=5.0, window=33.0, save_path=None):
    """Pe и Pm каждого СГ (zoom)."""
    fig, ax = _single_fig(w=12)
    tz, mask = _zoom_mask(res, event_time, window)
    for i in range(n_sg):
        ax.plot(tz, res['P_sg_each'][i][mask] / 1000, color=_SG_COLORS[i],
                lw=1.5, label=f'Pe СГ-{i+1}')
        ax.plot(tz, res['P_mech'][i][mask] / 1000, color=_SG_COLORS[i],
                lw=1.0, ls='--', alpha=0.6, label=f'Pm СГ-{i+1}')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('P, МВт')
    ax.set_title('Pe и Pm каждого СГ', loc='left', fontsize=10)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    _save(fig, save_path)


def plot_zoom_b(res, event_time=5.0, window=33.0, save_path=None):
    """Мощность ВСГ (zoom)."""
    fig, ax = _single_fig(w=12)
    tz, mask = _zoom_mask(res, event_time, window)
    ax.plot(tz, res['P_vsg'][mask] / 1000, color=_C['vsg'], lw=2)
    ax.axhline(0, color='0.7', lw=0.4, ls=':')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    pv = res['P_vsg'][mask] / 1000
    idx_max = np.argmax(np.abs(pv))
    ax.annotate(f'P_max = {pv[idx_max]:.2f} МВт',
                xy=(tz[idx_max], pv[idx_max]),
                xytext=(tz[idx_max] + 1.0, pv[idx_max] * 0.85),
                fontsize=8, color=_C['vsg'],
                arrowprops=dict(arrowstyle='->', color=_C['vsg'], lw=0.8))
    ax.set_xlabel('Время, с')
    ax.set_ylabel('P ВСГ, МВт')
    ax.set_title('Мощность ВСГ - агрессивный отклик', loc='left', fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)


def plot_zoom_c(res, event_time=5.0, window=33.0, save_path=None):
    """Частота (zoom)."""
    fig, ax = _single_fig(w=12)
    tz, mask = _zoom_mask(res, event_time, window)
    ax.plot(tz, res['f'][mask], color=_C['freq'], lw=1.5, label='f шины')
    f_vsg = res['omega_vsg'][mask] / (2 * np.pi)
    ax.plot(tz, f_vsg, color=_C['vsg'], lw=1.2, ls='--', label='f ВСГ')
    ax.axhline(50.0, color='0.7', lw=0.4, ls=':')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('f, Гц')
    ax.set_title('Частота - ВСГ опережает/запаздывает', loc='left', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)


def plot_zoom_d(res, n_sg, event_time=5.0, window=33.0, save_path=None):
    """Напряжение шины и клемм СГ (zoom)."""
    fig, ax = _single_fig(w=12)
    tz, mask = _zoom_mask(res, event_time, window)
    ax.plot(tz, res['V'][mask], color=_C['volt'], lw=1.5, label='V шины')
    for i in range(n_sg):
        ax.plot(tz, res['Vt_sg'][i][mask], color=_SG_COLORS[i],
                lw=0.8, alpha=0.6, label=f'Vt СГ-{i+1}')
    ax.axhline(1.0, color='0.7', lw=0.4, ls=':')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('V, о.е.')
    ax.set_title('Напряжение шины и клемм СГ', loc='left', fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, save_path)


def plot_zoom_e(res, n_sg, event_time=5.0, window=33.0, save_path=None):
    """АРВ: напряжение возбуждения (zoom)."""
    fig, ax = _single_fig(w=12)
    tz, mask = _zoom_mask(res, event_time, window)
    for i in range(n_sg):
        ax.plot(tz, res['Efd'][i][mask], color=_SG_COLORS[i],
                lw=1.2, label=f'Efd СГ-{i+1}')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Efd, о.е.')
    ax.set_title('АРВ: напряжение возбуждения', loc='left', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)


def plot_zoom_f(res, event_time=5.0, window=33.0, save_path=None):
    """Дисбаланс мощности на шине (zoom)."""
    fig, ax = _single_fig(w=12)
    tz, mask = _zoom_mask(res, event_time, window)
    P_gen = res['P_sg_total'][mask] + res['P_vsg'][mask]
    P_imb = (P_gen - res['P_load'][mask]) / 1000
    ax.plot(tz, P_imb, color=_C['imb'], lw=1.5)
    ax.fill_between(tz, 0, P_imb, where=(P_imb > 0), alpha=0.12, color=_C['vsg'])
    ax.fill_between(tz, 0, P_imb, where=(P_imb < 0), alpha=0.12, color=_C['load'])
    ax.axhline(0, color='0.7', lw=0.5, ls=':')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('ΔP, МВт')
    ax.set_title('Дисбаланс мощности на шине', loc='left', fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)


def _anti_mask(res, event_time, window):
    t = res['t']
    t0, t1 = event_time - 0.5, event_time + window
    mask = (t >= t0) & (t <= t1)
    return t[mask], mask


def plot_anti_a(res, n_sg, vsg_p, event_time=5.0, window=33.0, save_path=None):
    """Мощности машин - не могут установиться."""
    fig, ax = _single_fig(w=12)
    tz, mask = _anti_mask(res, event_time, window)
    for i in range(n_sg):
        ax.plot(tz, res['P_sg_each'][i][mask] / 1000, color=_SG_COLORS[i],
                lw=1.5, label=f'P СГ-{i+1}')
    ax.plot(tz, res['P_vsg'][mask] / 1000, color=_C['vsg'], lw=2.0,
            label='P ВСГ', zorder=10)
    P_target = res['P_load'][mask] / (n_sg * 1000)
    ax.plot(tz, P_target, color='0.5', lw=1.0, ls=':', label='P_target (равном.)')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('P, МВт')
    ax.legend(fontsize=7.5, ncol=3)
    ax.set_title('Мощности машин - не могут установиться', loc='left', fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)


def plot_anti_b(res, n_sg, event_time=5.0, window=33.0, save_path=None):
    """Разность мощностей между парами СГ - перетоки."""
    fig, ax = _single_fig(w=12)
    tz, mask = _anti_mask(res, event_time, window)
    P1 = res['P_sg_each'][0][mask] / 1000
    P2 = res['P_sg_each'][1][mask] / 1000
    P3 = res['P_sg_each'][2][mask] / 1000 if n_sg > 2 else np.zeros_like(P1)
    ax.plot(tz, P1 - P2, color='#7570B3', lw=1.5, label='P(СГ1) − P(СГ2)')
    if n_sg > 2:
        ax.plot(tz, P1 - P3, color='#E6AB02', lw=1.5, label='P(СГ1) − P(СГ3)')
        ax.plot(tz, P2 - P3, color='#66A61E', lw=1.5, label='P(СГ2) − P(СГ3)')
    ax.axhline(0, color='0.7', lw=0.5, ls=':')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('dP, МВт')
    ax.legend(fontsize=8)
    ax.set_title('Разность мощностей между парами СГ - перетоки',
                 loc='left', fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)


def plot_anti_c(res, n_sg, event_time=5.0, window=33.0, save_path=None):
    """ВСГ перехватывает мощность - СГ не успевают."""
    fig, ax = _single_fig(w=12)
    tz, mask = _anti_mask(res, event_time, window)
    ax.plot(tz, res['P_vsg'][mask] / 1000, color=_C['vsg'], lw=2.0, label='P ВСГ')
    ax.plot(tz, res['P_sg_total'][mask] / 1000, color=_C['sg'], lw=1.5, label='ΣP СГ')
    ax.plot(tz, res['P_load'][mask] / 1000, color=_C['load'], lw=1.5, ls='--',
            label='P нагрузки')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('P, МВт')
    ax.legend(fontsize=8)
    ax.set_title('ВСГ перехватывает мощность - СГ не успевают',
                 loc='left', fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)


# СЦЕНАРИЙ: ВСГ РАСКАЧИВАЕТ КУСТ ИЗ 3 СГ

if __name__ == "__main__":
    print("=" * 70)
    print(" СЦЕНАРИЙ: 3xСГ + ВСГ — АСИММЕТРИЯ ДЕМПФИРОВАНИЯ СГ-1 vs СГ-2")
    print(" СГ1 (D_mech=0.15, ramp=0.07) и СГ2 (D_mech=1.0, ramp=0.14) = 95%")
    print(" СГ3 = 0% (холостой ход на шине)")
    print(" Наброс -> ВСГ раскачивает -> СГ1 раскачивается, СГ2 демпфирован")
    print("=" * 70)

    N_SG = 3


    # ОДИНАКОВЫЕ СГ, но с испорченными параметрами для раскачки:
    #   - низкое демпфирование D_mech=0.5
    #   - медленный губернатор T_gov=1.2
    #   - малый ramp_up=0.10

    sg_common = SGParams(
        P_nom=1600.0, S_nom=2000.0,
        P_max=0.80,
        H=3.5, D_mech=0.5,          # <<< НИЗКОЕ демпфирование (было 8.0)
        Xd=1.2, Xq=0.8, Xd_prime=0.30, Xd_dprime=0.18,
        Td0_prime=5.0, Td0_dprime=0.035,
        R_droop=0.025,
        T_act=0.05,
        T_gov=1.2,                   # <<< МЕДЛЕННЫЙ губернатор (было 0.60)
        ramp_up=0.10, ramp_down=0.10,  # <<< МЕДЛЕННЫЙ набор (было 0.30)
        Ka=50.0, Ta=0.01, Te=1.5,
        Vref=1.0, Efd_max=5.0,
        X_line_sg=0.04, R_line_sg=0.005,
    )

    sg_list_params = []
    for i in range(N_SG):
        sp = deepcopy(sg_common)
        sp.name = f"СГ-{i + 1}"
        sg_list_params.append(sp)


    # АСИММЕТРИЯ СГ-1 vs СГ-2 (ТОЛЬКО механические параметры)
    # D_mech: разная амплитуда колебаний
    # ramp_up: разная скорость набора мощности -> фазовый сдвиг при переходных
    # H, T_gov, X_line - ОДИНАКОВЫЕ (иначе ломается резонанс с ВСГ)

    sg_list_params[0].D_mech = 0.45     # подобрано для выравнивания амплитуд
    sg_list_params[0].ramp_up = 0.07    # было 0.10 - медленнее набирает

    sg_list_params[1].D_mech = 0.80     # подобрано для выравнивания амплитуд
    sg_list_params[1].ramp_up = 0.14    # было 0.10 - быстрее набирает

    sg_list_params[2].D_mech = 0.75     # подобрано для выравнивания амплитуд


    # Начальная загрузка: СГ1,СГ2 = 95%, СГ3 = 0 (холостой ход)

    P0_each = [0.95 * 1600.0 / 2000.0,  # СГ1: 0.76 о.е. = 1520 кВт
               0.95 * 1600.0 / 2000.0,  # СГ2: 0.76 о.е. = 1520 кВт
               0.0]                       # СГ3: 0 (холостой ход)

    P0_load_kW = sum(P0_each[i] * sg_list_params[i].S_nom for i in range(N_SG))
    Q0_load_kvar = P0_load_kW * 0.3


    # Наброс нагрузки: один ступенчатый

    P_step_kW = 560.0
    P_after = P0_load_kW + P_step_kW  # 3600 кВт

    # Новая уставка - равная для всех 3 СГ (диспетчер выравнивает)
    P_new_each = P_after / 3.0 / 2000.0  # 0.60 о.е. = 1200 кВт = 75% P_nom
    P_new_each = min(P_new_each, 0.80)

    P_set_schedule = [
        (5.0, [P_new_each, P_new_each, P_new_each]),
    ]


    # ВСГ — АГРЕССИВНЫЕ настройки, провоцирующие раскачку

    vsg = VSGParams(
        P_nom=1600.0, S_nom=2000.0,
        P_max=0.80,
        H_virt=0.3,       # <<< Малая виртуальная инерция (было 0.5)
        D_virt=2.0,        # <<< Низкое демпфирование (было 15.0)
        R_droop=0.02,
        K_rocof=2.0,       # <<< Сильный отклик на df/dt (было 1.0)
        T_rocof=0.05,      # <<< Быстрый фильтр (было 0.10)
        K_ff=1.5,          # <<< Перекомпенсация (было 1.0)
        T_inv=0.005, zeta_inv=0.5,  # <<< Низкое демпфирование инвертора (было 0.65)
        I_max=1.2,
        Kp_v=25.0, Ki_v=400.0, T_qv=0.005,
        Xv=0.35,           # <<< Большой виртуальный импеданс (было 0.25)
        E_batt_kWh=1280.0,
        SoC_init=0.80,
        T_restore=0.0,
        T_washout=0.0,     # <<< Без washout — DC feedforward (было 1.5)
    )


    # Шина

    bus = BusParams(
        f_nom=50.0, V_nom=1.0,
        X_line=0.10,       # <<< Слабее связь (было 0.08)
        R_line=0.012,
        D_load=0.8,        # <<< Слабее зависимость нагрузки от f (было 1.0)
    )


    # Профиль нагрузки

    load = LoadProfile(events=[
        (0.0, P0_load_kW, Q0_load_kvar),          # 3040 кВт - начальная
        (5.0, P_after, P_after * 0.5),              # +560 -> 3600 кВт - наброс
    ])

    sim = SimParams(dt=0.001, T_end=40.0, downsample=5)

    # Вывод параметров

    print(f"\nСуммарная установленная: {N_SG * 1600 / 1000:.1f} МВт СГ + "
          f"{vsg.P_nom / 1000:.1f} МВт ВСГ = {(N_SG * 1600 + vsg.P_nom) / 1000:.1f} МВт")

    print(f"\nПараметры СГ (асимметрия D_mech и ramp_up):")
    for i, sp in enumerate(sg_list_params):
        print(f"  {sp.name}: H={sp.H}с, D_mech={sp.D_mech}, "
              f"T_gov={sp.T_gov}с, ramp_up={sp.ramp_up}, X_line_sg={sp.X_line_sg}")

    print(f"\nНачальная загрузка:")
    for i, sp in enumerate(sg_list_params):
        p_kw = P0_each[i] * sp.S_nom
        print(f"  {sp.name}: P0 = {P0_each[i] * 100:.0f}% S_nom = {p_kw:.0f} кВт "
              f"({p_kw / sp.P_nom * 100:.0f}% P_nom)")
    print(f"  P0_load = {P0_load_kW:.0f} кВт")

    print(f"\nНаброс при t=5с: +{P_step_kW:.0f} кВт -> P_load = {P_after:.0f} кВт")
    print(f"  Новая уставка каждого СГ: {P_new_each:.3f} о.е. = {P_new_each * 2000:.0f} кВт "
          f"({P_new_each * 2000 / 1600 * 100:.0f}% P_nom)")
    print(f"  СГ1,2: СКИДЫВАЮТ с {P0_each[0] * 2000:.0f} -> {P_new_each * 2000:.0f} кВт")
    print(f"  СГ3:   НАБИРАЕТ  с 0 -> {P_new_each * 2000:.0f} кВт")

    print(f"\nВСГ (АГРЕССИВНЫЕ настройки):")
    print(f"  H_virt={vsg.H_virt}с, D_virt={vsg.D_virt}, K_rocof={vsg.K_rocof}, "
          f"K_ff={vsg.K_ff}, Xv={vsg.Xv}, T_washout={vsg.T_washout}")


    # СИМУЛЯЦИЯ

    print(f"\n{'='*55}")
    print(f"Запуск симуляции...")
    print(f"{'='*55}")

    results = run_simulation(
        N_SG, sg_list_params, vsg, bus, load, sim,
        P0_sg_pu=P0_each,
        P_set_schedule=P_set_schedule
    )

    # Результаты

    print(f"\n{'='*55}")
    print(f"РЕЗУЛЬТАТЫ:")
    print(f"{'='*55}")
    print(f"  f:   {results['f'].min():.3f} .. {results['f'].max():.3f} Гц "
          f"(df_max = {np.abs(results['f'] - 50).max():.3f} Гц)")
    print(f"  V:   {results['V'].min():.4f} .. {results['V'].max():.4f} о.е.")
    print(f"  ВСГ: P_max = {results['P_vsg'].max():.0f} кВт "
          f"({results['P_vsg'].max()/vsg.P_nom*100:.0f}%)")

    for i in range(N_SG):
        P_arr = results['P_sg_each'][i]
        print(f"  {sg_list_params[i].name}: Pe: {P_arr.min():.0f} .. {P_arr.max():.0f} кВт "
              f"(размах = {P_arr.max()-P_arr.min():.0f} кВт)")

    print(f"\n  Анализ колебаний (t > 6с)")
    t_res = results['t']
    mask_after = t_res > 6.0
    for i in range(N_SG):
        P_i = results['P_sg_each'][i][mask_after]
        P_mean = P_i.mean()
        P_std = P_i.std()
        pct = P_std / max(P_mean, 1) * 100
        print(f"  СГ-{i+1}: P_mean={P_mean:.0f} кВт, P_std={P_std:.0f} кВт ({pct:.1f}%)")

    P_vsg_after = results['P_vsg'][mask_after]
    print(f"  ВСГ: P_mean={P_vsg_after.mean():.0f} кВт, P_std={P_vsg_after.std():.0f} кВт")
    print(f"\n  SoC: {results['SoC'][0]*100:.1f}% -> {results['SoC'][-1]*100:.1f}%")

    # Графики

    output_dir = ''
    print(f"\nСоздание графиков (13 отдельных файлов)...")

    ev = 5.0
    win = 33.0

    # Showcase (4)
    plot_showcase_a(results, N_SG, sg_list_params, vsg,
                    save_path=os.path.join(output_dir, "01_P_machines.png"))
    plot_showcase_b(results, vsg,
                    save_path=os.path.join(output_dir, "02_frequency.png"))
    plot_showcase_c(results, N_SG,
                    save_path=os.path.join(output_dir, "03_dP_deviations.png"))
    plot_showcase_d(results,
                    save_path=os.path.join(output_dir, "04_voltage.png"))

    # Zoom (6)
    plot_zoom_a(results, N_SG, event_time=ev, window=win,
                save_path=os.path.join(output_dir, "05_zoom_Pe_Pm.png"))
    plot_zoom_b(results, event_time=ev, window=win,
                save_path=os.path.join(output_dir, "06_zoom_P_vsg.png"))
    plot_zoom_c(results, event_time=ev, window=win,
                save_path=os.path.join(output_dir, "07_zoom_frequency.png"))
    plot_zoom_d(results, N_SG, event_time=ev, window=win,
                save_path=os.path.join(output_dir, "08_zoom_voltage.png"))
    plot_zoom_e(results, N_SG, event_time=ev, window=win,
                save_path=os.path.join(output_dir, "09_zoom_Efd.png"))
    plot_zoom_f(results, event_time=ev, window=win,
                save_path=os.path.join(output_dir, "10_zoom_imbalance.png"))

    # Antiphase (3)
    plot_anti_a(results, N_SG, vsg, event_time=ev, window=win,
                save_path=os.path.join(output_dir, "11_anti_P_machines.png"))
    plot_anti_b(results, N_SG, event_time=ev, window=win,
                save_path=os.path.join(output_dir, "12_anti_dP_pairs.png"))
    plot_anti_c(results, N_SG, event_time=ev, window=win,
                save_path=os.path.join(output_dir, "13_anti_VSG_vs_SG.png"))

    plt.close('all')
    print(f"\n{'='*55}")
    print(f"ГОТОВО! Графики сохранены в {output_dir}")
    print(f"{'='*55}")