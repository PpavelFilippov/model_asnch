import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List
from copy import deepcopy

@dataclass
class SGParams:
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
    osc_enabled: bool = False
    osc_gain: float = 0.0
    osc_freq: float = 2.0
    osc_start: float = 5.0
    osc_phase: float = 0.0

@dataclass
class BusParams:
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
        (20.0, 500.0, 250.0),
        (35.0, 900.0, 450.0),
        (50.0, 600.0, 300.0),
    ])

@dataclass
class SimParams:
    dt: float = 0.001
    T_end: float = 60.0
    downsample: int = 10

class SynchronousGenerator:
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
        dEq_p = (self.Efd - self.Eq_prime - (p.Xd - p.Xd_prime) * self.Id) / p.Td0_prime
        self.Eq_prime += dEq_p * dt
        dEq_dp = (self.Eq_prime - self.Eq_dprime - (p.Xd_prime - p.Xd_dprime) * self.Id) / p.Td0_dprime
        self.Eq_dprime += dEq_dp * dt
        return self.Pe * self.p.S_nom, self.Qe * self.p.S_nom

class VirtualSynchronousGenerator:
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
        self.t_sim = 0.0

    def init_steady_state(self, P0_pu, Q0_pu, V_bus):
        self.P_out = P0_pu
        self.P_ref = P0_pu
        self.P_set = P0_pu
        self.Q_out = Q0_pu
        self.Q_ref = Q0_pu
        self.omega_virt = self.w0
        self.delta_virt = 0.0
        self.V_int_err = Q0_pu / (self.p.Ki_v + 1e-9)
        self.t_sim = 0.0

    def step(self, dt, V_bus, f_bus):
        p = self.p
        w0 = self.w0
        self.t_sim += dt
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
        P_osc = 0.0
        if p.osc_enabled and self.t_sim >= p.osc_start:
            P_osc = p.osc_gain * np.sin(2 * np.pi * p.osc_freq * (self.t_sim - p.osc_start) + p.osc_phase)
        dw_pu = (self.omega_virt - w0) / w0
        P_total_ref = self.P_set + P_droop + P_rocof + P_ff + P_osc
        P_accel = P_total_ref - self.P_out - p.D_virt * dw_pu
        if p.H_virt > 0.01:
            d_omega = (P_accel / (2.0 * p.H_virt)) * w0 * dt
        else:
            d_omega = 0.0
        self.omega_virt += d_omega
        self.delta_virt += (self.omega_virt - w0) * dt
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
        if (self.Q_out >= Q_max_avail and V_error > 0) or (self.Q_out <= -Q_max_avail and V_error < 0):
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

class BusModel:
    def __init__(self, params: BusParams):
        self.p = params
        self.f = params.f_nom
        self.V = params.V_nom
        self.w0 = 2 * np.pi * params.f_nom
        self.P_vsg_prev = 0.0

    def step(self, dt, P_gen_total_kW, Q_gen_total_kvar, P_load_base_kW, Q_load_base_kvar, H_total, S_total_kVA, omega_coi, P_vsg_kW=0.0, Q_vsg_kvar=0.0, Xv_vsg=0.1):
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
        V_target = (p.V_nom + p.R_line * dP_pu + p.X_line * Q_imbalance_pu + V_emf_transient)
        T_v = 0.02
        self.V += (V_target - self.V) / T_v * dt
        self.V = np.clip(self.V, 0.85, 1.15)
        return self.f, self.V, P_load_kW, Q_load_kvar

def get_load_at_time(t, load_profile: LoadProfile):
    P, Q = load_profile.events[0][1], load_profile.events[0][2]
    for ev_t, ev_P, ev_Q in load_profile.events:
        if t >= ev_t:
            P, Q = ev_P, ev_Q
        else:
            break
    return P, Q

def run_simulation(n_sg: int = 3, sg_params=None, vsg_params: VSGParams = None, bus_params: BusParams = None, load_profile: LoadProfile = None, sim_params: SimParams = None, P0_sg_pu: list = None, P_set_schedule: list = None):
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
    H_total = (sum(sp.H * sp.S_nom for sp in sg_params_list) + vsg_params.H_virt * vsg_params.S_nom) / S_total
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
        'omega_sg': np.zeros((n_sg, n_out)),
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
        f_bus, V_bus, P_load_actual, Q_load_actual = bus.step(dt, P_gen_total, Q_gen_total, P_load_base, Q_load_base, H_total, S_total, omega_coi, P_vsg_kW=P_vsg_kW, Q_vsg_kvar=Q_vsg_kvar, Xv_vsg=vsg_params.Xv)
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
                out['omega_sg'][i, oi] = sg.omega
            oi += 1
    for key in out:
        if isinstance(out[key], np.ndarray):
            if out[key].ndim == 1:
                out[key] = out[key][:oi]
            else:
                out[key] = out[key][:, :oi]
    return out

def plot_results(res, n_sg, sg_p, vsg_p, save_path=None):
    t = res['t']
    fig = plt.figure(figsize=(16, 22))
    fig.suptitle(f'Параллельная работа {n_sg}xСГ ({sg_p.P_nom} кВт, H={sg_p.H}с) + ВСГ ({vsg_p.P_nom} кВт, H_virt={vsg_p.H_virt}с)\nСГ: ramp^={sg_p.ramp_up * sg_p.P_nom:.0f} кВт/с, droop={sg_p.R_droop * 100:.0f}%, T_gov={sg_p.T_gov}с  |  ВСГ: D={vsg_p.D_virt}, droop={vsg_p.R_droop * 100:.0f}%, K_rocof={vsg_p.K_rocof}, K_ff={vsg_p.K_ff}, tau_inv={vsg_p.T_inv * 1000:.0f}мс, BESS={vsg_p.E_batt_kWh} кВт*ч', fontsize=12, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(6, 2, hspace=0.45, wspace=0.3, left=0.07, right=0.96, top=0.94, bottom=0.03)
    C_SG = '#2166AC'
    C_VSG = '#1B7837'
    C_LOAD = '#D6604D'
    C_FREQ = '#7B3294'
    C_VOLT = '#E08214'
    C_sg_each = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#B2182B']
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(t, 0, res['P_sg_total'], alpha=0.2, color=C_SG)
    ax.fill_between(t, res['P_sg_total'], res['P_sg_total'] + np.maximum(res['P_vsg'], 0), alpha=0.2, color=C_VSG)
    ax.plot(t, res['P_sg_total'], color=C_SG, lw=1.5, label='СГ суммарно')
    ax.plot(t, res['P_vsg'], color=C_VSG, lw=2, label='ВСГ (BESS)')
    ax.plot(t, res['P_load'], color=C_LOAD, lw=2, ls='--', label='Нагрузка')
    ax.plot(t, res['P_sg_total'] + res['P_vsg'], color='gray', lw=1, ls=':', alpha=0.6, label='Генерация суммарно')
    ax.set_ylabel('P, кВт')
    ax.set_title('Распределение активной мощности')
    ax.legend(loc='upper left', fontsize=9, ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax = fig.add_subplot(gs[1, 0])
    for i in range(n_sg):
        c = C_sg_each[i % len(C_sg_each)]
        ax.plot(t, res['P_sg_each'][i], color=c, lw=1.5, label=f'Pe СГ-{i + 1}')
        ax.plot(t, res['P_mech'][i], color=c, lw=1, ls='--', alpha=0.6, label=f'Pm СГ-{i + 1}')
    ax.set_ylabel('P, кВт')
    ax.set_title('Мощности СГ (сплошная - электр., пунктир - мех.)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, res['P_vsg'], color=C_VSG, lw=2, label='P_ВСГ')
    ax.plot(t, res['Q_vsg'], color='#66C2A5', lw=1.5, ls='--', label='Q_ВСГ')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axhline(vsg_p.P_nom, color=C_VSG, lw=0.5, ls=':', alpha=0.4)
    ax.axhline(-vsg_p.P_nom * 0.5, color='red', lw=0.5, ls=':', alpha=0.4)
    ax.set_ylabel('кВт / квар')
    ax.set_title('Мощность ВСГ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, res['f'], color=C_FREQ, lw=2)
    ax.axhline(50.0, color='gray', lw=0.5, ls='--')
    ax.fill_between(t, 49.8, 50.2, alpha=0.05, color='green')
    f_dev = np.abs(res['f'] - 50.0)
    idx = np.argmax(f_dev)
    ax.annotate(f'df_max = {f_dev[idx]:.3f} Гц', xy=(t[idx], res['f'][idx]), xytext=(t[idx] + 2, res['f'][idx] + 0.03 * np.sign(res['f'][idx] - 50)), fontsize=8, color='red', arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    ax.set_ylabel('f, Гц')
    ax.set_title('Частота шины')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, res['V'], color=C_VOLT, lw=2, label='V_шина')
    for i in range(min(n_sg, 3)):
        ax.plot(t, res['Vt_sg'][i], color=C_sg_each[i % len(C_sg_each)], lw=0.8, alpha=0.5, label=f'Vt СГ-{i + 1}')
    ax.axhline(1.0, color='gray', lw=0.5, ls='--')
    ax.fill_between(t, 0.95, 1.05, alpha=0.05, color='green')
    ax.set_ylabel('V, о.е.')
    ax.set_title('Напряжение (шина + клеммы СГ)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t, res['SoC'] * 100, color='#1B9E77', lw=2)
    ax.axhline(vsg_p.SoC_min * 100, color='red', lw=1, ls='--', label=f'SoC_min={vsg_p.SoC_min * 100:.0f}%')
    ax.axhline(vsg_p.SoC_max * 100, color='blue', lw=1, ls='--', label=f'SoC_max={vsg_p.SoC_max * 100:.0f}%')
    ax.set_ylabel('SoC, %')
    ax.set_xlabel('Время, с')
    ax.set_title(f'Батарея ВСГ ({vsg_p.E_batt_kWh} кВт*ч)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 100)
    ax = fig.add_subplot(gs[3, 1])
    ax.plot(t, res['Q_sg_total'], color=C_SG, lw=1.5, label='Q_СГ')
    ax.plot(t, res['Q_vsg'], color=C_VSG, lw=1.5, label='Q_ВСГ')
    ax.plot(t, res['Q_load'], color=C_LOAD, lw=1.5, ls='--', label='Q_нагрузка')
    ax.set_ylabel('Q, квар')
    ax.set_xlabel('Время, с')
    ax.set_title('Реактивная мощность')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax = fig.add_subplot(gs[4, 0])
    for i in range(n_sg):
        ax.plot(t, res['Efd'][i], color=C_sg_each[i % len(C_sg_each)], lw=1.2, label=f'Efd СГ-{i + 1}')
    ax.set_ylabel('Efd, о.е.')
    ax.set_xlabel('Время, с')
    ax.set_title('Напряжение возбуждения (АРВ)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax = fig.add_subplot(gs[4, 1])
    f_vsg = res['omega_vsg'] / (2 * np.pi)
    ax.plot(t, res['f'], color=C_FREQ, lw=1.5, label='f_шина')
    ax.plot(t, f_vsg, color=C_VSG, lw=1.5, ls='--', label='f_ВСГ (виртуальная)')
    ax.axhline(50.0, color='gray', lw=0.5, ls='--')
    ax.set_ylabel('f, Гц')
    ax.set_xlabel('Время, с')
    ax.set_title('Частота шины vs виртуальная частота ВСГ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax = fig.add_subplot(gs[5, 0])
    for i in range(n_sg):
        f_sg = res['omega_sg'][i] / (2 * np.pi)
        ax.plot(t, f_sg, color=C_sg_each[i % len(C_sg_each)], lw=1.5, label=f'f_СГ-{i + 1}')
    ax.plot(t, res['f'], color='gray', lw=1.0, ls=':', label='f_шина')
    ax.axhline(50.0, color='gray', lw=0.5, ls='--')
    ax.set_ylabel('f, Гц')
    ax.set_xlabel('Время, с')
    ax.set_title('Частоты роторов СГ (расхождение = межмашинные колебания)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax = fig.add_subplot(gs[5, :])
    ax.plot(t, res['P_load'], color=C_LOAD, lw=2, label='P_нагрузка (с учётом f)')
    ax.plot(t, res['Q_load'], color='#FC8D62', lw=1.5, ls='--', label='Q_нагрузка')
    ax.set_ylabel('кВт / квар')
    ax.set_xlabel('Время, с')
    ax.set_title('Профиль нагрузки (с зависимостью P от частоты)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def plot_inter_machine_oscillations(res, n_sg, save_path=None):
    t = res['t']
    t0, t1 = 5.0, 20.0
    m = (t >= t0) & (t <= t1)
    tz = t[m]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('МЕЖМАШИННЫЕ КОЛЕБАНИЯ: СГ в противофазе', fontsize=14, fontweight='bold')
    ax = axes[0]
    colors = ['#2166AC', '#D6604D', '#1B7837']
    for i in range(n_sg):
        ax.plot(tz, res['P_sg_each'][i][m], color=colors[i], lw=2.5, label=f'СГ-{i + 1} Pe')
        ax.plot(tz, res['P_mech'][i][m], color=colors[i], lw=1.5, ls='--', alpha=0.5, label=f'СГ-{i + 1} Pm')
    ax.set_ylabel('P, кВт', fontsize=11)
    ax.set_title('Активная мощность каждого СГ (видно противофазность)', fontsize=12)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(tz[0], tz[-1])
    ax = axes[1]
    if n_sg >= 2:
        P_diff_12 = res['P_sg_each'][0][m] - res['P_sg_each'][1][m]
        ax.plot(tz, P_diff_12 / 1000, color='#7B3294', lw=2.5, label='СГ1 − СГ2')
    if n_sg >= 3:
        P_diff_23 = res['P_sg_each'][1][m] - res['P_sg_each'][2][m]
        ax.plot(tz, P_diff_23 / 1000, color='#E08214', lw=2.5, label='СГ2 − СГ3')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_ylabel('DP, МВт', fontsize=11)
    ax.set_title('Разность мощностей (амплитуда колебаний между машинами)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(tz[0], tz[-1])
    ax = axes[2]
    for i in range(n_sg):
        f_sg = res['omega_sg'][i][m] / (2 * np.pi)
        ax.plot(tz, f_sg, color=colors[i], lw=2, label=f'f_СГ-{i + 1}')
    ax.plot(tz, res['f'][m], color='gray', lw=1.5, ls=':', label='f_шина')
    ax.axhline(50.0, color='gray', lw=0.5, ls='--')
    ax.set_ylabel('f, Гц', fontsize=11)
    ax.set_xlabel('Время, с', fontsize=11)
    ax.set_title('Частоты роторов СГ (расхождение = межмашинные колебания)', fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(tz[0], tz[-1])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_transient_zoom(res, n_sg, sg_p, vsg_p, event_time=5.0, window=3.0, save_path=None):
    t = res['t']
    t0 = event_time - 0.5
    t1 = event_time + window
    mask = (t >= t0) & (t <= t1)
    tz = t[mask]
    C_SG = '#2166AC'
    C_VSG = '#1B7837'
    C_LOAD = '#D6604D'
    C_FREQ = '#7B3294'
    C_VOLT = '#E08214'
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(f'ZOOM: переходный процесс при набросе нагрузки (t = {event_time} с)\nВзаимодействие ВСГ  <-> куст СГ: перетоки P и Q, раскачка напряжения', fontsize=13, fontweight='bold')
    ax = axes[0, 0]
    ax.plot(tz, res['P_sg_total'][mask], color=C_SG, lw=2, label='P СГ (сумма)')
    ax.plot(tz, res['P_vsg'][mask], color=C_VSG, lw=2, label='P ВСГ')
    ax.plot(tz, res['P_load'][mask], color=C_LOAD, lw=2, ls='--', label='P нагрузка')
    ax.plot(tz, res['P_sg_total'][mask] + res['P_vsg'][mask], color='gray', lw=1, ls=':', label='Генерация')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('P, кВт')
    ax.set_title('Активная мощность')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    ax.plot(tz, res['Q_sg_total'][mask], color=C_SG, lw=2, label='Q СГ (сумма)')
    ax.plot(tz, res['Q_vsg'][mask], color=C_VSG, lw=2, label='Q ВСГ')
    ax.plot(tz, res['Q_load'][mask], color=C_LOAD, lw=2, ls='--', label='Q нагрузка')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('Q, квар')
    ax.set_title('Реактивная мощность - циркуляция Q')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    ax.plot(tz, res['V'][mask], color=C_VOLT, lw=2, label='V шина')
    ax.axhline(1.0, color='gray', lw=0.5, ls='--')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('V, о.е.')
    ax.set_title('Напряжение шины - толчок от ВСГ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    ax.plot(tz, res['f'][mask], color=C_FREQ, lw=2, label='f шина')
    f_vsg = res['omega_vsg'][mask] / (2 * np.pi)
    ax.plot(tz, f_vsg, color=C_VSG, lw=1.5, ls='--', label='f ВСГ (вирт.)')
    ax.axhline(50.0, color='gray', lw=0.5, ls='--')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('f, Гц')
    ax.set_title('Частота')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[2, 0]
    ax.plot(tz, res['P_vsg'][mask], color=C_VSG, lw=2.5, label='P ВСГ')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    pv = res['P_vsg'][mask]
    idx_max = np.argmax(pv)
    ax.annotate(f'overshoot: {pv[idx_max]:.0f} кВт', xy=(tz[idx_max], pv[idx_max]), xytext=(tz[idx_max] + 0.3, pv[idx_max] * 0.9), fontsize=9, color=C_VSG, arrowprops=dict(arrowstyle='->', color=C_VSG, lw=0.8))
    ax.set_ylabel('P, кВт')
    ax.set_title('Мощность ВСГ - перерегулирование инвертора')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[2, 1]
    for i in range(min(n_sg, 3)):
        ax.plot(tz, res['Efd'][i][mask], lw=1.5, label=f'Efd СГ-{i + 1}')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('Efd, о.е.')
    ax.set_title('АРВ: реакция на возмущение V от ВСГ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[3, 0]
    ax.plot(tz, res['P_sg_each'][0][mask], color=C_SG, lw=2, label='Pe СГ-1')
    ax.plot(tz, res['P_mech'][0][mask], color=C_SG, lw=1.5, ls='--', label='Pm СГ-1')
    ax.fill_between(tz, res['P_sg_each'][0][mask], res['P_mech'][0][mask], alpha=0.15, color=C_SG)
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('P, кВт')
    ax.set_xlabel('Время, с')
    ax.set_title('СГ-1: Pe vs Pm (заливка = dP -> изменение omega)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[3, 1]
    P_gen_total = res['P_sg_total'][mask] + res['P_vsg'][mask]
    P_imb = P_gen_total - res['P_load'][mask]
    ax.plot(tz, P_imb, color='#E7298A', lw=2, label='P_gen − P_load')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.fill_between(tz, 0, P_imb, alpha=0.15, color='#E7298A')
    ax.set_ylabel('dP, кВт')
    ax.set_xlabel('Время, с')
    ax.set_title('Дисбаланс мощности на шине')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def plot_all_machines_zoom(res, n_sg, sg_p, vsg_p, event_time=5.0, window=4.0, save_path=None):
    t = res['t']
    t0 = event_time - 0.5
    t1 = event_time + window
    mask = (t >= t0) & (t <= t1)
    tz = t[mask]
    C_VSG = '#1B7837'
    sg_colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F4A582', '#D6604D', '#B2182B', '#762A83', '#5AAE61']
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'ВСЕ МАШИНЫ: переходный процесс при набросе нагрузки (t = {event_time} с)\n{n_sg}xСГ + 1xВСГ - распределение P и Q между генераторами', fontsize=13, fontweight='bold')
    ax = axes[0, 0]
    for i in range(n_sg):
        ax.plot(tz, res['P_sg_each'][i][mask] / 1000, color=sg_colors[i % len(sg_colors)], lw=1.2, label=f'СГ-{i + 1}')
    ax.plot(tz, res['P_vsg'][mask] / 1000, color=C_VSG, lw=2.5, label='ВСГ', zorder=10)
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('P, МВт')
    ax.set_title('Активная мощность каждой машины')
    ax.legend(fontsize=7, ncol=5, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    for i in range(n_sg):
        ax.plot(tz, res['Q_sg_each'][i][mask] / 1000, color=sg_colors[i % len(sg_colors)], lw=1.2, label=f'СГ-{i + 1}')
    ax.plot(tz, res['Q_vsg'][mask] / 1000, color=C_VSG, lw=2.5, label='ВСГ', zorder=10)
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('Q, Мвар')
    ax.set_title('Реактивная мощность каждой машины')
    ax.legend(fontsize=7, ncol=5, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    P_stack = np.zeros_like(tz)
    for i in range(n_sg):
        P_next = P_stack + res['P_sg_each'][i][mask] / 1000
        ax.fill_between(tz, P_stack, P_next, alpha=0.4, color=sg_colors[i % len(sg_colors)], label=f'СГ-{i + 1}')
        P_stack = P_next
    P_vsg_mw = res['P_vsg'][mask] / 1000
    P_next = P_stack + np.maximum(P_vsg_mw, 0)
    ax.fill_between(tz, P_stack, P_next, alpha=0.6, color=C_VSG, label='ВСГ')
    ax.plot(tz, res['P_load'][mask] / 1000, color='red', lw=2, ls='--', label='Нагрузка')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.3)
    ax.set_ylabel('P, МВт')
    ax.set_xlabel('Время, с')
    ax.set_title('Stacked: вклад каждой машины в покрытие нагрузки')
    ax.legend(fontsize=7, ncol=6, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    idx_base = np.searchsorted(tz, event_time - 0.1)
    if idx_base >= len(tz):
        idx_base = 0
    for i in range(n_sg):
        P_i = res['P_sg_each'][i][mask]
        dP_i = (P_i - P_i[idx_base]) / 1000
        ax.plot(tz, dP_i, color=sg_colors[i % len(sg_colors)], lw=1.2, label=f'dP СГ-{i + 1}')
    P_vsg_arr = res['P_vsg'][mask]
    dP_vsg = (P_vsg_arr - P_vsg_arr[idx_base]) / 1000
    ax.plot(tz, dP_vsg, color=C_VSG, lw=2.5, label='dP ВСГ', zorder=10)
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('dP, МВт')
    ax.set_xlabel('Время, с')
    ax.set_title('Отклонение P от базы - кто и как реагирует на наброс')
    ax.legend(fontsize=7, ncol=5, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

if __name__ == "__main__":
    N_SG = 3
    sg_common = SGParams(
        P_nom=1600.0, S_nom=2000.0,
        P_max=0.80,
        H=3.5, D_mech=2.0,
        Xd=1.2, Xq=0.8, Xd_prime=0.30, Xd_dprime=0.18,
        Td0_prime=5.0, Td0_dprime=0.035,
        R_droop=0.025,
        T_act=0.05,
        T_gov=0.60,
        ramp_up=0.30, ramp_down=0.30,
        Ka=50.0, Ta=0.01, Te=1.5,
        Vref=1.0, Efd_max=5.0,
        X_line_sg=0.12,
        R_line_sg=0.02,
    )
    sg_list_params = []
    sg1 = deepcopy(sg_common)
    sg1.name = "СГ-1"
    sg1.H = 2.5
    sg1.D_mech = 0.8
    sg1.T_gov = 0.25
    sg1.R_droop = 0.03
    sg_list_params.append(sg1)
    sg2 = deepcopy(sg_common)
    sg2.name = "СГ-2"
    sg2.H = 5.5
    sg2.D_mech = 0.6
    sg2.T_gov = 1.0
    sg2.R_droop = 0.04
    sg_list_params.append(sg2)
    sg3 = deepcopy(sg_common)
    sg3.name = "СГ-3"
    sg3.H = 3.2
    sg3.D_mech = 0.5
    sg3.T_gov = 0.5
    sg3.R_droop = 0.035
    sg_list_params.append(sg3)
    P0_each = [0.50, 0.50, 0.0]
    P0_load_kW = sum(P0_each[i] * sg_list_params[i].S_nom for i in range(N_SG))
    Q0_load_kvar = P0_load_kW * 0.3
    P_step_kW = 560.0
    P_after = P0_load_kW + P_step_kW
    P_new_each = P_after / 3.0 / 2000.0
    P_new_each = min(P_new_each, 0.80)
    P_set_schedule = [(5.0, [P_new_each, P_new_each, P_new_each])]
    vsg = VSGParams(
        P_nom=1600.0, S_nom=2000.0,
        P_max=0.90,
        H_virt=0.25,
        D_virt=2.0,
        R_droop=0.015,
        K_rocof=3.0,
        T_rocof=0.05,
        K_ff=1.8,
        T_inv=0.0015,
        zeta_inv=0.4,
        I_max=1.5,
        Kp_v=50.0,
        Ki_v=500.0,
        T_qv=0.003,
        Xv=0.35,
        E_batt_kWh=1280.0,
        SoC_init=0.80,
        T_restore=0.0,
        T_washout=0.5,
        osc_enabled=True,
        osc_gain=0.12,
        osc_freq=1.8,
        osc_start=5.0,
        osc_phase=0.0,
    )
    bus = BusParams(
        f_nom=50.0, V_nom=1.0,
        X_line=0.08, R_line=0.008,
        D_load=1.0,
    )
    load = LoadProfile(events=[
        (0.0, P0_load_kW, Q0_load_kvar),
        (5.0, P_after, P_after * 0.5),
    ])
    sim = SimParams(dt=0.001, T_end=25.0, downsample=5)
    results = run_simulation(N_SG, sg_list_params, vsg, bus, load, sim, P0_sg_pu=P0_each, P_set_schedule=P_set_schedule)
    t_mask = (results['t'] >= 5.0) & (results['t'] <= 20.0)
    sg_avg = sg_list_params[0]
    plot_results(results, N_SG, sg_avg, vsg, save_path="sg_vsg_oscillations_full.png")
    plot_transient_zoom(results, N_SG, sg_avg, vsg, event_time=5.0, window=15.0, save_path="zoom_transient_5s.png")
    plot_all_machines_zoom(results, N_SG, sg_avg, vsg, event_time=5.0, window=15.0, save_path="all_machines_5s.png")
    plot_inter_machine_oscillations(results, N_SG, save_path="inter_machine_oscillations.png")
    plt.close('all')