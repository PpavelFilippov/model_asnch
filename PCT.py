"""
Графики для демонстративного отчёта.
Запускает симуляцию из TRY_TWO_v2.py и генерирует 13 отдельных графиков.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import os
import importlib.util

spec = importlib.util.spec_from_file_location("sim", "TRY_TWO_v2.py")
sim_module = importlib.util.module_from_spec(spec)

import sys
_argv = sys.argv
sys.argv = ['']

from io import StringIO
_stdout = sys.stdout
sys.stdout = StringIO()
spec.loader.exec_module(sim_module)
sys.stdout = _stdout
sys.argv = _argv

from copy import deepcopy

N_SG = 3

sg_common = sim_module.SGParams(
    P_nom=1600.0, S_nom=2000.0, P_max=0.80,
    H=3.5, D_mech=0.5, Xd=1.2, Xq=0.8, Xd_prime=0.30, Xd_dprime=0.18,
    Td0_prime=5.0, Td0_dprime=0.035, R_droop=0.025,
    T_act=0.05, T_gov=1.2, ramp_up=0.10, ramp_down=0.10,
    Ka=50.0, Ta=0.01, Te=1.5, Vref=1.0, Efd_max=5.0,
    X_line_sg=0.04, R_line_sg=0.005,
)

sg_list_params = []
for i in range(N_SG):
    sp = deepcopy(sg_common)
    sp.name = f"СГ-{i + 1}"
    sg_list_params.append(sp)

sg_list_params[0].D_mech = 0.45
sg_list_params[0].ramp_up = 0.07
sg_list_params[1].D_mech = 0.80
sg_list_params[1].ramp_up = 0.14
sg_list_params[2].D_mech = 0.75

P0_each = [0.95 * 1600.0 / 2000.0, 0.95 * 1600.0 / 2000.0, 0.0]
P0_load_kW = sum(P0_each[i] * sg_list_params[i].S_nom for i in range(N_SG))
Q0_load_kvar = P0_load_kW * 0.3

P_step_kW = 560.0
P_after = P0_load_kW + P_step_kW
P_new_each = min(P_after / 3.0 / 2000.0, 0.80)

P_set_schedule = [(5.0, [P_new_each, P_new_each, P_new_each])]

vsg = sim_module.VSGParams(
    P_nom=1600.0, S_nom=2000.0, P_max=0.80,
    H_virt=0.3, D_virt=2.0, R_droop=0.02,
    K_rocof=2.0, T_rocof=0.05, K_ff=1.5,
    T_inv=0.005, zeta_inv=0.5, I_max=1.2,
    Kp_v=25.0, Ki_v=400.0, T_qv=0.005,
    Xv=0.35, E_batt_kWh=1280.0, SoC_init=0.80,
    T_restore=0.0, T_washout=0.0,
)

bus = sim_module.BusParams(f_nom=50.0, V_nom=1.0, X_line=0.10, R_line=0.012, D_load=0.8)
load = sim_module.LoadProfile(events=[
    (0.0, P0_load_kW, Q0_load_kvar),
    (5.0, P_after, P_after * 0.5),
])
sim_p = sim_module.SimParams(dt=0.001, T_end=40.0, downsample=5)

print("Запуск симуляции...")
res = sim_module.run_simulation(
    N_SG, sg_list_params, vsg, bus, load, sim_p,
    P0_sg_pu=P0_each, P_set_schedule=P_set_schedule
)
print("Симуляция завершена.\n")

# СТИЛЬ

# Светлая палитра
BG       = '#FFFFFF'
BG_AX    = '#FFFFFF'
GRID_C   = '#E5E7EB'
TEXT_C   = '#000000'
TEXT_DIM = '#000000'
ACCENT   = '#3B82F6'

# Цвета машин
C_SG1  = '#DC2626'   # красный
C_SG2  = '#0891B2'   # тёмно-бирюзовый
C_SG3  = '#D97706'   # янтарный
C_VSG  = '#059669'   # изумрудный
C_FREQ = '#7C3AED'   # фиолетовый
C_VOLT = '#EA580C'   # оранжевый
C_EFD  = '#DB2777'   # розовый
C_IMB  = '#2563EB'   # синий
C_LOAD = '#6B7280'   # серый

SG_COLORS = [C_SG1, C_SG2, C_SG3]
SG_NAMES  = ['СГ-1', 'СГ-2', 'СГ-3']
SG_SHORT  = ['СГ-1', 'СГ-2', 'СГ-3']


def setup_style():
    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor': BG_AX,
        'axes.edgecolor': '#000000',
        'axes.labelcolor': TEXT_C,
        'axes.titlepad': 28,
        'text.color': TEXT_C,
        'xtick.color': '#000000',
        'ytick.color': '#000000',
        'grid.color': GRID_C,
        'grid.alpha': 0.7,
        'grid.linewidth': 0.4,
        'grid.linestyle': '-',
        'axes.grid': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 1.6,
        'lines.antialiased': True,
        'figure.dpi': 150,
        'savefig.dpi': 250,
        'savefig.facecolor': BG,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.facecolor': '#FFFFFF',
        'legend.edgecolor': '#000000',
        'legend.framealpha': 0.95,
        'legend.fancybox': True,
        'legend.borderpad': 0.8,
        'legend.labelspacing': 0.5,
    })


def make_fig(w=14, h=5):
    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    return fig, ax


def make_fig_with_zoom(w=14, h_main=5, h_zoom=3, ratio=None):
    """Фигура с двумя панелями: основной график сверху, zoom снизу."""
    setup_style()
    if ratio is None:
        ratio = [h_main, h_zoom]
    fig, (ax_main, ax_zoom) = plt.subplots(
        2, 1, figsize=(w, h_main + h_zoom),
        gridspec_kw={'height_ratios': ratio, 'hspace': 0.35}
    )
    return fig, ax_main, ax_zoom


def save_fig(fig, path):
    fig.savefig(path, dpi=250, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f" {os.path.basename(path)}")


def event_line(ax, t=5.0):
    ax.axvline(t, color='#DC2626', lw=1.0, ls='--', alpha=0.6, zorder=0)
    ax.text(t + 0.3, ax.get_ylim()[1] * 0.97, 'НАБРОС', color='#DC2626',
            fontsize=7.5, fontstyle='italic', alpha=0.7, va='top',
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])


def subtitle(ax, text):
    """Подзаголовок мелким шрифтом под основным заголовком."""
    ax.text(0.0, 1.02, text, transform=ax.transAxes,
            fontsize=8.5, color=TEXT_DIM, va='bottom')


def add_watermark(ax):
    ax.text(0.99, 0.02, '3xСГ + ВСГ  *  Изолированная энергосистема',
            transform=ax.transAxes, fontsize=7, color='#D1D5DB',
            ha='right', va='bottom', style='italic')


def connect_zoom(fig, ax_main, ax_zoom, t_start, t_end):
    """
    Рисует затенённую зону на основном графике и трапециевидные
    соединительные линии от неё к нижнему подграфику.
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, ConnectionPatch

    # Затенение области на основном графике
    ax_main.axvspan(t_start, t_end, alpha=0.08, color=ACCENT, zorder=0)

    # Левая соединительная линия: от (t_start, bottom основного) к (left, top zoom)
    con_left = ConnectionPatch(
        xyA=(t_start, ax_main.get_ylim()[0]), coordsA=ax_main.transData,
        xyB=(ax_zoom.get_xlim()[0], ax_zoom.get_ylim()[1]), coordsB=ax_zoom.transData,
        color='#000000', linewidth=0.8, linestyle='-', alpha=0.5)
    fig.add_artist(con_left)

    # Правая соединительная линия
    con_right = ConnectionPatch(
        xyA=(t_end, ax_main.get_ylim()[0]), coordsA=ax_main.transData,
        xyB=(ax_zoom.get_xlim()[1], ax_zoom.get_ylim()[1]), coordsB=ax_zoom.transData,
        color='#000000', linewidth=0.8, linestyle='-', alpha=0.5)
    fig.add_artist(con_right)


def fill_zoom_ax(ax_zoom, t_arr, datasets, colors, t_start, t_end,
                 lws=None, styles=None, alphas=None,
                 fill_between_data=None, ylabel=None, xlabel='Время, с'):
    """Заполняет нижний подграфик данными из указанного временного окна."""
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    t_z = t_arr[mask]

    for idx, (data, col) in enumerate(zip(datasets, colors)):
        lw = lws[idx] if lws else 1.6
        ls = styles[idx] if styles else '-'
        al = alphas[idx] if alphas else 0.95
        ax_zoom.plot(t_z, data[mask], color=col, lw=lw, ls=ls, alpha=al)

    if fill_between_data is not None:
        fb = fill_between_data
        d = fb['data'][mask]
        ax_zoom.fill_between(t_z, 0, d, where=(d > 0), alpha=0.18, color=fb.get('color_pos', C_VSG))
        ax_zoom.fill_between(t_z, 0, d, where=(d < 0), alpha=0.18, color=fb.get('color_neg', C_SG1))

    ax_zoom.set_xlim(t_start, t_end)
    ax_zoom.set_facecolor('#FAFBFC')
    ax_zoom.grid(True, alpha=0.5, linewidth=0.3)
    if ylabel:
        ax_zoom.set_ylabel(ylabel, fontsize=10)
    if xlabel:
        ax_zoom.set_xlabel(xlabel, fontsize=10)

    for spine in ax_zoom.spines.values():
        spine.set_edgecolor('#000000')

    return ax_zoom


# ДАННЫЕ

t = res['t']
EV = 5.0
WIN = 33.0

# zoom mask
t0z, t1z = EV - 1.0, EV + WIN
mz = (t >= t0z) & (t <= t1z)
tz = t[mz]

# anti mask
t0a, t1a = EV - 0.5, EV + WIN
ma = (t >= t0a) & (t <= t1a)
ta = t[ma]

OUT = 'output/'
os.makedirs(OUT, exist_ok=True)

# 01 - АКТИВНАЯ МОЩНОСТЬ ВСЕХ МАШИН

ZOOM1_S, ZOOM1_E = 25, 32

fig, ax, ax_z = make_fig_with_zoom(14, 5.0, 3.5)

for i in range(N_SG):
    ax.plot(t, res['P_sg_each'][i] / 1000, color=SG_COLORS[i],
            lw=1.8, label=SG_NAMES[i], alpha=0.92)
ax.plot(t, res['P_vsg'] / 1000, color=C_VSG, lw=2.2,
        label='ВСГ', zorder=10)

ax.axvspan(0, EV, alpha=0.04, color=ACCENT, zorder=0)
ax.text(2.5, ax.get_ylim()[0] + 0.05, 'СТАЦИОНАРНЫЙ\nРЕЖИМ',
        fontsize=8, color=ACCENT, alpha=0.4, ha='center', va='bottom')

event_line(ax, EV)
ax.axhline(0, color=GRID_C, lw=0.5, ls=':')
ax.set_ylabel('Активная мощность P, МВт')
ax.set_title('Активная мощность генераторов', fontweight='bold', fontsize=15, color=TEXT_C)
ax.legend(loc='upper right', ncol=2)
ax.set_xlim(t[0], t[-1])
ax.set_xticklabels([])

fill_zoom_ax(ax_z, t,
             [res['P_sg_each'][0]/1000, res['P_sg_each'][1]/1000,
              res['P_sg_each'][2]/1000, res['P_vsg']/1000],
             [C_SG1, C_SG2, C_SG3, C_VSG],
             ZOOM1_S, ZOOM1_E,
             lws=[1.8, 1.8, 1.4, 1.8], ylabel='P, МВт')

connect_zoom(fig, ax, ax_z, ZOOM1_S, ZOOM1_E)
#add_watermark(ax_z)
save_fig(fig, f'{OUT}/01_P_machines.png')


# 02 - ЧАСТОТА ШИНЫ

ZOOM2_S, ZOOM2_E = 20, 28

fig, ax, ax_z = make_fig_with_zoom(14, 4.5, 3.5)

f_vsg = res['omega_vsg'] / (2 * np.pi)
ax.plot(t, res['f'], color=C_FREQ, lw=2.0, label='Частота шины', zorder=5)
ax.plot(t, f_vsg, color=C_VSG, lw=1.2, ls='--', alpha=0.7,
        label='Виртуальная частота ВСГ')
ax.axhline(50.0, color=TEXT_DIM, lw=0.5, ls=':', alpha=0.5)
ax.fill_between(t, 49.8, 50.2, alpha=0.05, color=C_VSG, zorder=0)

event_line(ax, EV)
ax.set_ylabel('Частота f, Гц')
ax.set_title('Частота на шине', fontweight='bold', fontsize=15, color=TEXT_C)
ax.legend(loc='upper right')
ax.set_xlim(t[0], t[-1])
ax.set_xticklabels([])

fill_zoom_ax(ax_z, t,
             [res['f'], f_vsg],
             [C_FREQ, C_VSG],
             ZOOM2_S, ZOOM2_E,
             lws=[2.0, 1.0], styles=['-', '--'], alphas=[1.0, 0.6],
             ylabel='f, Гц')

connect_zoom(fig, ax, ax_z, ZOOM2_S, ZOOM2_E)
#add_watermark(ax_z)
save_fig(fig, f'{OUT}/02_frequency.png')


# 03 - ОТКЛОНЕНИЯ dP ОТ СРЕДНЕГО

ZOOM3_S, ZOOM3_E = 28, 35

fig, ax, ax_z = make_fig_with_zoom(14, 4.5, 3.5)

P_sg_mean = np.mean([res['P_sg_each'][i] for i in range(N_SG)], axis=0)
dP_arrays = [(res['P_sg_each'][i] - P_sg_mean) / 1000 for i in range(N_SG)]
for i in range(N_SG):
    ax.plot(t, dP_arrays[i], color=SG_COLORS[i], lw=1.6,
            label=f'dP {SG_SHORT[i]}', alpha=0.9)

ax.axhline(0, color=TEXT_DIM, lw=0.5, ls=':')
event_line(ax, EV)
ax.set_ylabel('dP (от среднего), МВт')
ax.set_title('Отклонения мощности от среднего', fontweight='bold', fontsize=15, color=TEXT_C)
ax.legend(loc='upper right')
ax.set_xlim(t[0], t[-1])
ax.set_xticklabels([])

fill_zoom_ax(ax_z, t, dP_arrays, SG_COLORS,
             ZOOM3_S, ZOOM3_E, lws=[1.6, 1.6, 1.4], ylabel='ΔP, МВт')

connect_zoom(fig, ax, ax_z, ZOOM3_S, ZOOM3_E)
#add_watermark(ax_z)
save_fig(fig, f'{OUT}/03_dP_deviations.png')


# 04 - НАПРЯЖЕНИЕ ШИНЫ

ZOOM4_S, ZOOM4_E = 25, 33

fig, ax, ax_z = make_fig_with_zoom(14, 4.0, 3.0)

ax.plot(t, res['V'], color=C_VOLT, lw=2.0, label='Напряжение шины')
ax.axhline(1.0, color=TEXT_DIM, lw=0.5, ls=':')
ax.fill_between(t, 0.95, 1.05, alpha=0.06, color=C_VSG, zorder=0)
ax.text(1, 1.05, 'Нормальный диапазон +-5%', fontsize=7, color=C_VSG, alpha=0.6, va='bottom')

event_line(ax, EV)
ax.set_ylabel('Напряжение V, о.е.')
ax.set_title('Напряжение на шине', fontweight='bold', fontsize=15, color=TEXT_C)
ax.legend(loc='upper right')
ax.set_xlim(t[0], t[-1])
ax.set_xticklabels([])

fill_zoom_ax(ax_z, t, [res['V']], [C_VOLT],
             ZOOM4_S, ZOOM4_E, lws=[2.0], ylabel='V, о.е.')

connect_zoom(fig, ax, ax_z, ZOOM4_S, ZOOM4_E)
#add_watermark(ax_z)
save_fig(fig, f'{OUT}/04_voltage.png')


# 05 - ZOOM Pe и Pm каждого СГ

fig, ax, ax_z = make_fig_with_zoom(14, 5.0, 3.5)

for i in range(N_SG):
    ax.plot(tz, res['P_sg_each'][i][mz] / 1000, color=SG_COLORS[i],
            lw=1.8, label=f'Pe {SG_SHORT[i]}', alpha=0.9)
    ax.plot(tz, res['P_mech'][i][mz] / 1000, color=SG_COLORS[i],
            lw=1.0, ls=':', alpha=0.45, label=f'Pm {SG_SHORT[i]}')

event_line(ax, EV)
ax.set_ylabel('P, МВт')
ax.set_title('Электрическая (Pe) и механическая (Pm) мощность', fontweight='bold',
             fontsize=15, color=TEXT_C)
ax.legend(fontsize=7.5, ncol=3, loc='upper right')
ax.set_xticklabels([])

Z5_S, Z5_E = 5, 10
fill_zoom_ax(ax_z, tz,
             [res['P_sg_each'][0][mz]/1000, res['P_mech'][0][mz]/1000,
              res['P_sg_each'][1][mz]/1000, res['P_mech'][1][mz]/1000],
             [C_SG1, C_SG1, C_SG2, C_SG2],
             Z5_S, Z5_E,
             lws=[1.8, 1.0, 1.8, 1.0],
             styles=['-', ':', '-', ':'],
             alphas=[0.9, 0.45, 0.9, 0.45],
             ylabel='P, МВт')

connect_zoom(fig, ax, ax_z, Z5_S, Z5_E)
#add_watermark(ax_z)
save_fig(fig, f'{OUT}/05_zoom_Pe_Pm.png')


# 06 - ZOOM мощность ВСГ

fig, ax, ax_z = make_fig_with_zoom(14, 4.5, 3.0)

pv = res['P_vsg'][mz] / 1000
ax.fill_between(tz, 0, pv, where=(pv > 0), alpha=0.15, color=C_VSG, zorder=0)
ax.fill_between(tz, 0, pv, where=(pv < 0), alpha=0.15, color=C_SG1, zorder=0)
ax.plot(tz, pv, color=C_VSG, lw=2.2)

idx_max = np.argmax(pv)
ax.annotate(f'P_max = {pv[idx_max]:.2f} МВт',
            xy=(tz[idx_max], pv[idx_max]),
            xytext=(tz[idx_max] + 3, pv[idx_max] * 0.82),
            fontsize=9, color=C_VSG, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_VSG, lw=1.2),
            path_effects=[pe.withStroke(linewidth=2.5, foreground=BG_AX)])

ax.axhline(1.28, color=C_VSG, lw=0.6, ls=':', alpha=0.4)
ax.text(tz[-1], 1.30, '80% P_nom', fontsize=7, color=C_VSG, alpha=0.5, ha='right')
ax.axhline(0, color=TEXT_DIM, lw=0.5, ls=':')
event_line(ax, EV)
ax.set_ylabel('P ВСГ, МВт')
ax.set_title('Мощность ВСГ', fontweight='bold', fontsize=15, color=TEXT_C)
ax.set_xticklabels([])

Z6_S, Z6_E = 5, 9
fill_zoom_ax(ax_z, tz, [pv], [C_VSG],
             Z6_S, Z6_E, lws=[2.2], ylabel='P ВСГ, МВт',
             fill_between_data={'data': pv, 'color_pos': C_VSG, 'color_neg': C_SG1})

connect_zoom(fig, ax, ax_z, Z6_S, Z6_E)
#add_watermark(ax_z)
save_fig(fig, f'{OUT}/06_zoom_P_vsg.png')


# 07 - ZOOM частота

fig, ax = make_fig(14, 5)

ax.plot(tz, res['f'][mz], color=C_FREQ, lw=2.0, label='Частота шины', zorder=5)
f_vsg_z = res['omega_vsg'][mz] / (2 * np.pi)
ax.plot(tz, f_vsg_z, color=C_VSG, lw=1.0, ls='--', alpha=0.6,
        label='Виртуальная частота ВСГ')
ax.axhline(50.0, color=TEXT_DIM, lw=0.5, ls=':')
ax.fill_between(tz, 49.8, 50.2, alpha=0.05, color=C_VSG, zorder=0)

event_line(ax, EV)
ax.set_xlabel('Время, с')
ax.set_ylabel('f, Гц')
ax.set_title('Частота - ВСГ опережает/запаздывает', fontweight='bold',
             fontsize=15, color=TEXT_C)
ax.legend(loc='upper right')
#add_watermark(ax)
save_fig(fig, f'{OUT}/07_zoom_frequency.png')


# 08 - ZOOM напряжение + Vt клемм

fig, ax = make_fig(14, 5)

ax.plot(tz, res['V'][mz], color=C_VOLT, lw=2.2, label='V шины', zorder=5)
for i in range(N_SG):
    ax.plot(tz, res['Vt_sg'][i][mz], color=SG_COLORS[i],
            lw=0.9, alpha=0.55, label=f'Vt {SG_SHORT[i]}')
ax.axhline(1.0, color=TEXT_DIM, lw=0.5, ls=':')
ax.fill_between(tz, 0.95, 1.05, alpha=0.05, color=C_VSG, zorder=0)

event_line(ax, EV)
ax.set_xlabel('Время, с')
ax.set_ylabel('V, о.е.')
ax.set_title('Напряжение шины и клемм генераторов', fontweight='bold',
             fontsize=15, color=TEXT_C)
ax.legend(loc='upper right', ncol=2)
#add_watermark(ax)
save_fig(fig, f'{OUT}/08_zoom_voltage.png')


# 09 - ZOOM Efd (АРВ)

fig, ax = make_fig(14, 5)

for i in range(N_SG):
    ax.plot(tz, res['Efd'][i][mz], color=SG_COLORS[i],
            lw=1.6, label=f'Efd {SG_SHORT[i]}', alpha=0.9)

event_line(ax, EV)
ax.set_xlabel('Время, с')
ax.set_ylabel('Efd, о.е.')
ax.set_title('АРВ: напряжение возбуждения', fontweight='bold',
             fontsize=15, color=TEXT_C)
ax.legend(loc='upper right')
#add_watermark(ax)
save_fig(fig, f'{OUT}/09_zoom_Efd.png')


# 10 - ZOOM дисбаланс мощности

fig, ax, ax_z = make_fig_with_zoom(14, 4.5, 3.0)

P_gen = res['P_sg_total'][mz] + res['P_vsg'][mz]
P_imb = (P_gen - res['P_load'][mz]) / 1000

ax.fill_between(tz, 0, P_imb, where=(P_imb > 0), alpha=0.2, color=C_VSG, zorder=0,
                label='Избыток генерации')
ax.fill_between(tz, 0, P_imb, where=(P_imb < 0), alpha=0.2, color=C_SG1, zorder=0,
                label='Дефицит генерации')
ax.plot(tz, P_imb, color=C_IMB, lw=1.8)
ax.axhline(0, color=TEXT_DIM, lw=0.8, ls='-', alpha=0.3)

event_line(ax, EV)
ax.set_ylabel('dP (ген − нагр), МВт')
ax.set_title('Дисбаланс мощности на шине', fontweight='bold', fontsize=15, color=TEXT_C)
ax.legend(loc='upper right')
ax.set_xticklabels([])

Z10_S, Z10_E = 30, 37
fill_zoom_ax(ax_z, tz, [P_imb], [C_IMB],
             Z10_S, Z10_E, lws=[1.8], ylabel='dP, МВт',
             fill_between_data={'data': P_imb, 'color_pos': C_VSG, 'color_neg': C_SG1})

connect_zoom(fig, ax, ax_z, Z10_S, Z10_E)
#add_watermark(ax_z)
save_fig(fig, f'{OUT}/10_zoom_imbalance.png')


# 11 - ANTIPHASE: мощности машин + P_target

fig, ax = make_fig(14, 5.5)

for i in range(N_SG):
    ax.plot(ta, res['P_sg_each'][i][ma] / 1000, color=SG_COLORS[i],
            lw=1.6, label=f'P {SG_SHORT[i]}', alpha=0.9)
ax.plot(ta, res['P_vsg'][ma] / 1000, color=C_VSG, lw=2.2,
        label='P ВСГ', zorder=10)

P_target = res['P_load'][ma] / (N_SG * 1000)
ax.plot(ta, P_target, color=TEXT_DIM, lw=1.2, ls=':', alpha=0.5,
        label='Целевая P (равномерная)')

event_line(ax, EV)
ax.set_xlabel('Время, с')
ax.set_ylabel('P, МВт')
ax.set_title('Мощности генераторов - невозможность установиться', fontweight='bold',
             fontsize=15, color=TEXT_C)
ax.legend(loc='upper right', ncol=3, fontsize=8)
#add_watermark(ax)
save_fig(fig, f'{OUT}/11_anti_P_machines.png')


# 12 - ANTIPHASE: ΔP между парами

fig, ax = make_fig(14, 5)

P1 = res['P_sg_each'][0][ma] / 1000
P2 = res['P_sg_each'][1][ma] / 1000
P3 = res['P_sg_each'][2][ma] / 1000

ax.plot(ta, P1 - P2, color='#A78BFA', lw=1.8, label='P(СГ-1) − P(СГ-2)')
ax.plot(ta, P1 - P3, color='#FBBF24', lw=1.6, label='P(СГ-1) − P(СГ-3)', alpha=0.85)
ax.plot(ta, P2 - P3, color='#34D399', lw=1.6, label='P(СГ-2) − P(СГ-3)', alpha=0.85)

ax.axhline(0, color=TEXT_DIM, lw=0.5, ls=':')
event_line(ax, EV)
ax.set_xlabel('Время, с')
ax.set_ylabel('dP, МВт')
ax.set_title('Перетоки мощности между генераторами', fontweight='bold',
             fontsize=15, color=TEXT_C)
ax.legend(loc='upper right')
#add_watermark(ax)
save_fig(fig, f'{OUT}/12_anti_dP_pairs.png')


# 13 - ANTIPHASE: ВСГ vs суммарный СГ

fig, ax, ax_z = make_fig_with_zoom(14, 5.0, 3.5)

ax.plot(ta, res['P_vsg'][ma] / 1000, color=C_VSG, lw=2.2, label='P ВСГ')
ax.plot(ta, res['P_sg_total'][ma] / 1000, color=ACCENT, lw=1.8, label='ΣP всех СГ')
ax.plot(ta, res['P_load'][ma] / 1000, color=C_LOAD, lw=1.8, ls='--',
        label='P нагрузки')

sg_tot = res['P_sg_total'][ma] / 1000
p_ld = res['P_load'][ma] / 1000
ax.fill_between(ta, sg_tot, p_ld, where=(p_ld > sg_tot),
                alpha=0.1, color=C_VSG, zorder=0, label='Компенсация ВСГ')

event_line(ax, EV)
ax.set_ylabel('P, МВт')
ax.set_title('ВСГ перехватывает мощность - СГ не успевают', fontweight='bold',
             fontsize=15, color=TEXT_C)
ax.legend(loc='upper right', ncol=2)
ax.set_xticklabels([])

Z13_S, Z13_E = 6, 12
fill_zoom_ax(ax_z, ta,
             [res['P_vsg'][ma]/1000, res['P_sg_total'][ma]/1000, res['P_load'][ma]/1000],
             [C_VSG, ACCENT, C_LOAD],
             Z13_S, Z13_E,
             lws=[2.2, 1.8, 1.8],
             styles=['-', '-', '--'],
             alphas=[1.0, 1.0, 0.7],
             ylabel='P, МВт')

connect_zoom(fig, ax, ax_z, Z13_S, Z13_E)
#add_watermark(ax_z)
save_fig(fig, f'{OUT}/13_anti_VSG_vs_SG.png')


# 14 - РЕАКТИВНАЯ МОЩНОСТЬ КАЖДОГО ГЕНЕРАТОРА

ZOOM14_S, ZOOM14_E = 25, 32

fig, ax, ax_z = make_fig_with_zoom(14, 5.0, 3.5)

for i in range(N_SG):
    ax.plot(t, res['Q_sg_each'][i] / 1000, color=SG_COLORS[i],
            lw=1.8, label=SG_NAMES[i], alpha=0.92)
ax.plot(t, res['Q_vsg'] / 1000, color=C_VSG, lw=2.2,
        label='ВСГ', zorder=10)

ax.axhline(0, color=GRID_C, lw=0.5, ls=':')
event_line(ax, EV)
ax.set_ylabel('Реактивная мощность Q, Мвар')
ax.set_title('Реактивная мощность генераторов', fontweight='bold', fontsize=15, color=TEXT_C)
ax.legend(loc='upper right', ncol=2)
ax.set_xlim(t[0], t[-1])
ax.set_xticklabels([])

fill_zoom_ax(ax_z, t,
             [res['Q_sg_each'][0]/1000, res['Q_sg_each'][1]/1000,
              res['Q_sg_each'][2]/1000, res['Q_vsg']/1000],
             [C_SG1, C_SG2, C_SG3, C_VSG],
             ZOOM14_S, ZOOM14_E,
             lws=[1.8, 1.8, 1.4, 1.8], ylabel='Q, Мвар')

connect_zoom(fig, ax, ax_z, ZOOM14_S, ZOOM14_E)
#add_watermark(ax_z)
save_fig(fig, f'{OUT}/14_Q_machines.png')


# 15 - Q ВСГ ОТДЕЛЬНО (zoom)

fig, ax, ax_z = make_fig_with_zoom(14, 4.5, 3.0)

qv = res['Q_vsg'][mz] / 1000
ax.fill_between(tz, 0, qv, where=(qv > 0), alpha=0.15, color=C_VSG, zorder=0)
ax.fill_between(tz, 0, qv, where=(qv < 0), alpha=0.15, color=C_SG1, zorder=0)
ax.plot(tz, qv, color=C_VSG, lw=2.2)

ax.axhline(0, color=TEXT_C, lw=0.5, ls=':')
event_line(ax, EV)
ax.set_ylabel('Q ВСГ, Мвар')
ax.set_title('Реактивная мощность ВСГ', fontweight='bold', fontsize=15, color=TEXT_C)
ax.set_xticklabels([])

Z15_S, Z15_E = 5, 10
fill_zoom_ax(ax_z, tz, [qv], [C_VSG],
             Z15_S, Z15_E, lws=[2.2], ylabel='Q ВСГ, Мвар',
             fill_between_data={'data': qv, 'color_pos': C_VSG, 'color_neg': C_SG1})

connect_zoom(fig, ax, ax_z, Z15_S, Z15_E)
#add_watermark(ax_z)
save_fig(fig, f'{OUT}/15_zoom_Q_vsg.png')


# 16 - БАЛАНС Q: суммарная генерация vs нагрузка

fig, ax, ax_z = make_fig_with_zoom(14, 5.0, 3.5)

Q_sg_tot = res['Q_sg_total'][ma] / 1000
Q_vsg_a = res['Q_vsg'][ma] / 1000
Q_gen_tot = Q_sg_tot + Q_vsg_a
Q_load_a = res['Q_load'][ma] / 1000

ax.plot(ta, Q_sg_tot, color=ACCENT, lw=1.8, label='ΣQ всех СГ')
ax.plot(ta, Q_vsg_a, color=C_VSG, lw=2.2, label='Q ВСГ')
ax.plot(ta, Q_load_a, color=C_LOAD, lw=1.8, ls='--', label='Q нагрузки')
ax.axhline(0, color=TEXT_C, lw=0.5, ls=':')

event_line(ax, EV)
ax.set_ylabel('Q, Мвар')
ax.set_title('Баланс реактивной мощности', fontweight='bold', fontsize=15, color=TEXT_C)
ax.legend(loc='upper right', ncol=3)
ax.set_xticklabels([])

Z16_S, Z16_E = 6, 12
fill_zoom_ax(ax_z, ta,
             [Q_sg_tot, Q_vsg_a, Q_load_a],
             [ACCENT, C_VSG, C_LOAD],
             Z16_S, Z16_E,
             lws=[1.8, 2.2, 1.8],
             styles=['-', '-', '--'],
             alphas=[1.0, 1.0, 0.7],
             ylabel='Q, Мвар')

connect_zoom(fig, ax, ax_z, Z16_S, Z16_E)
#add_watermark(ax_z)
save_fig(fig, f'{OUT}/16_Q_balance.png')


print(f"\n{'═' * 50}")
print(f"Все 16 графиков сохранены в {OUT}")
print(f"{'═' * 50}")
