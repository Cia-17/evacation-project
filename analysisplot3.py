import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import os

os.makedirs("results", exist_ok=True)


#  SIMULATE EVACUATION 
def run_sim(n, speed, panic, blocked):
    
    rng = np.random.default_rng(42)
    cx, cy = 350, 290

    # place people in oval stands
    r   = 0.45 + rng.random(n) * 0.48
    ang = rng.random(n) * 2 * np.pi
    x   = cx + r * 240 * np.cos(ang) + rng.normal(0, 8, n)
    y   = cy + r * 185 * np.sin(ang) + rng.normal(0, 8, n)

    # exit positions
    e_ang  = np.radians([90,45,0,315,270,225,180,135])
    ex     = cx + 280 * np.cos(e_ang)
    ey     = cy - 220 * np.sin(e_ang)
    open_e = [i for i in range(8) if i not in blocked]

    # assign each person their nearest open exit
    target = np.zeros(n, dtype=int)
    for i in range(n):
        dists     = [np.hypot(ex[j]-x[i], ey[j]-y[i]) for j in open_e]
        target[i] = open_e[np.argmin(dists)]

    gx   = ex[target]
    gy   = ey[target]
    spds = (0.9 + rng.random(n) * 0.8) * speed * 0.02
    out  = np.zeros(n, dtype=bool)

    rows        = []
    FPS_SIM     = 60
    max_q       = np.zeros(8, dtype=int)
    wait_sum    = np.zeros(8)
    wait_cnt    = np.zeros(8, dtype=int)

    # tracking crowd density
    PIXELS_PER_METER = 700/105   #stadium pixel size/over standard stadium size

    ZONE_RADIUS = 8
    ZONE_RADIUS_M = ZONE_RADIUS / PIXELS_PER_METER
    ZONE_AREA = np.pi * ZONE_RADIUS_M**2   
    peak_density = np.zeros(8)

    for tick in range(FPS_SIM * 500):
        alive = ~out
        dx    = gx[alive] - x[alive]
        dy    = gy[alive] - y[alive]
        dist  = np.hypot(dx, dy)

        arrived   = dist < 6
        alive_idx = np.where(alive)[0]
        out[alive_idx[arrived]] = True

        if tick % FPS_SIM == 0:
            rows.append((tick // FPS_SIM, int(out.sum())))

        if out.all():
            rows.append((tick // FPS_SIM, int(out.sum())))
            break

        # check queues + density once per second
        if tick % 60 == 0:
            for i in open_e:
                dists_to_exit = np.hypot(ex[i] - x[alive_idx], ey[i] - y[alive_idx])

                # queue 
                near = np.sum(dists_to_exit < 40)
                if near > max_q[i]:
                    max_q[i] = near
                if near > 0:
                    wait_sum[i] += near
                    wait_cnt[i] += 1

                # density within the zone radius secified
                zone_count   = np.sum(dists_to_exit < ZONE_RADIUS)
                density_here = zone_count / ZONE_AREA
                if density_here > peak_density[i]:
                    peak_density[i] = density_here

        move  = ~arrived
        m_idx = alive_idx[move]

        density = np.zeros(len(m_idx))
        sample  = rng.choice(len(m_idx), min(50, len(m_idx)), replace=False)
        for si in sample:
            ddx         = x[m_idx] - x[m_idx[si]]
            ddy         = y[m_idx] - y[m_idx[si]]
            density[si] = np.sum(np.hypot(ddx, ddy) < 20)

        slowdown = 1 - np.clip(density / 80, 0, 0.4) * panic
        eff_spd  = spds[m_idx] * (0.5 + 0.5 * dist[move] / (dist[move] + 25)) * slowdown

        x[m_idx] += (dx[move] / dist[move]) * eff_spd
        y[m_idx] += (dy[move] / dist[move]) * eff_spd

    safe_cnt = np.where(wait_cnt > 0, wait_cnt, 1)
    avg_w    = np.where(wait_cnt > 0, wait_sum / safe_cnt / FPS_SIM * 10, 0)

    return pd.DataFrame(rows, columns=["second", "evacuated"]), max_q, avg_w, peak_density



# run all 3 scenarios
SCENARIOS = {
    "normal":  {"label": "Normal",        "blocked": [1,5],    "speed": 1.0, "panic": 1.0, "color": "#1B5F8C"},
    "partial": {"label": "Partial Block", "blocked": [1], "speed": 0.80,"panic": 1.3, "color": "#2A9DD4"},
    "panic":   {"label": "Panic",         "blocked": [], "speed": 0.50,"panic": 4.0, "color": "#C0392B"},
}
print("Running simulations...")
data         = {}
max_queue    = {}
avg_wait     = {}
peak_density = {}

for key, sc in SCENARIOS.items():
    print(f"  {sc['label']}...")
    df, mq, aw, pd_vals = run_sim(5000, sc["speed"], sc["panic"], sc["blocked"])
    df.to_csv(f"results/timeline_{key}.csv", index=False)
    data[key]         = df
    max_queue[key]    = mq.tolist()
    avg_wait[key]     = [round(v, 1) for v in aw.tolist()]
    peak_density[key] = [round(v, 2) for v in pd_vals.tolist()]

print("Done.\n")



#  MILESTONE TIMES 


def get_milestones(df, total=5000):
    ms = {}
    for pct in [25, 50, 75, 90, 100]:
        need = total * pct / 100
        row  = df[df["evacuated"] >= need]
        ms[pct] = int(row["second"].iloc[0]) if not row.empty else None
    return ms

milestones = {k: get_milestones(data[k]) for k in SCENARIOS}

# save milestone table
rows = []
for k, ms in milestones.items():
    for pct, t in ms.items():
        rows.append({"scenario": SCENARIOS[k]["label"], "milestone": f"{pct}%", "time_s": t})
pd.DataFrame(rows).to_csv("results/milestones.csv", index=False)


#  LOGISTIC MODEL + R² + RMSE


def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

fits      = {}
fit_stats = {}   # R² and RMSE per scenario

for key, df in data.items():
    time   = df["second"].values.astype(float)
    actual = df["evacuated"].values.astype(float)
    mid    = time[len(time) // 2]

    # split: train on first 80%, test on last 20%
    split      = int(0.8 * len(time))
    train_time = time[:split];   train_actual = actual[:split]
    test_time  = time[split:];   test_actual  = actual[split:]

    try:
        best, _ = curve_fit(
            logistic, train_time, train_actual,
            p0=[5000, 0.05, mid],
            bounds=([4000,0.001,0], [5200,1,500]),
            maxfev=10000
        )
        fits[key] = best

        # R² on full data
        predicted  = logistic(time, *best)
        ss_res     = np.sum((actual - predicted) ** 2)
        ss_tot     = np.sum((actual - actual.mean()) ** 2)
        r_squared  = round(1 - ss_res / ss_tot, 4)

        # RMSE on held-out 20%
        test_pred  = logistic(test_time, *best)
        rmse       = round(np.sqrt(np.mean((test_actual - test_pred) ** 2)), 1)

        fit_stats[key] = {"r2": r_squared, "rmse": rmse}

    except Exception:
        fits[key]      = None
        fit_stats[key] = {"r2": None, "rmse": None}


#  SENSITIVITY ANALYSIS (fast: 3 extra sims)
# vary panic factor for the panic scenario — low, base, high


print("Running sensitivity analysis...")

panic_low  = run_sim(5000, 0.60, 1.4, [1,5])[0]   # panic factor reduced
panic_base = data["panic"]                          # original panic scenario
panic_high = run_sim(5000, 0.60, 25, [1,5])[0]   # panic factor increased

sens_labels = ["Low panic (1.4)", "Base panic (4.0)", "High panic (25)"]
sens_data   = [panic_low, panic_base, panic_high]
sens_colors = ["#5BA85A", "#C0392B", "#7D1A0F"]

print("Done.\n")


exit_names = ["N","NE","E","SE","S","SW","W","NW"]


#  first plot: EVACUATION TIMELINE + LOGISTIC FIT


fig1 = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Cumulative evacuation over time",
                    "Logistic model fit (curve_fit)"]
)

for key, sc in SCENARIOS.items():
    df = data[key]

    fig1.add_trace(go.Scatter(
        x=df["second"], y=df["evacuated"],
        name=sc["label"],
        line=dict(color=sc["color"], width=2),
        mode="lines"
    ), row=1, col=1)

    for pct in [50, 90]:
        t = milestones[key].get(pct)
        if t:
            fig1.add_trace(go.Scatter(
                x=[t], y=[5000 * pct / 100],
                mode="markers+text",
                marker=dict(color=sc["color"], size=9),
                text=[f"{pct}%"],
                textposition="top right",
                textfont=dict(size=9),
                showlegend=False
            ), row=1, col=1)

    if fits[key] is not None:
        K, r, t0 = fits[key]
        st       = fit_stats[key]
        t_smooth = np.linspace(0, df["second"].max(), 200)
        y_smooth = logistic(t_smooth, K, r, t0)
        fig1.add_trace(go.Scatter(
            x=t_smooth, y=y_smooth,
            name=f"{sc['label']} (fit  R²={st['r2']}  RMSE={st['rmse']})",
            line=dict(color=sc["color"], dash="dash", width=1.5)
        ), row=1, col=2)
        fig1.add_trace(go.Scatter(
            x=df["second"], y=df["evacuated"],
            mode="markers",
            marker=dict(color=sc["color"], size=4, opacity=0.4),
            showlegend=False
        ), row=1, col=2)

fig1.add_hline(y=5000, line_dash="dot", line_color="gray", annotation_text="Full (5000)", row=1, col=1)
fig1.update_xaxes(title_text="Time (seconds)")
fig1.update_yaxes(title_text="People evacuated")
fig1.update_layout(title="Evacuation Timeline & Logistic Fit", height=500, template="plotly_white")
fig1.write_html("results/plot1_timeline.html")
print("Saved plot1_timeline.html")


#  second plot: MILESTONE BAR CHART


fig2 = go.Figure()
for key, sc in SCENARIOS.items():
    ms     = milestones[key]
    labels = ["25%","50%","75%","90%","100%"]
    times  = [ms.get(p, 0) for p in [25,50,75,90,100]]
    fig2.add_trace(go.Bar(
        name=sc["label"], x=labels, y=times,
        marker_color=sc["color"],
        text=[f"{t}s" for t in times],
        textposition="outside"
    ))

fig2.add_hline(y=180, line_dash="dash", line_color="red", annotation_text="180s safety target")
fig2.update_layout(
    title=" Time to Clear Each Milestone",
    xaxis_title="Milestone", yaxis_title="Time (seconds)",
    barmode="group", height=480, template="plotly_white"
)
fig2.write_html("results/plot2_milestones.html")
print("Saved plot2_milestones.html")


# third plot: BOTTLENECK: MAX QUEUE DEPTH PER EXIT


fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Maximum queue per exit", "Average wait time per exit (s)"]
)

for key, sc in SCENARIOS.items():
    fig3.add_trace(go.Bar(
        name=sc["label"], x=exit_names, y=max_queue[key],
        marker_color=sc["color"]
    ), row=1, col=1)
    fig3.add_trace(go.Bar(
        name=sc["label"], x=exit_names, y=avg_wait[key],
        marker_color=sc["color"], showlegend=False
    ), row=1, col=2)

fig3.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Critical level", row=1, col=1)
fig3.update_layout(title="Bottleneck Analysis", height=480, template="plotly_white", barmode="group")
fig3.write_html("results/plot3_bottleneck.html")
print("Saved plot3_bottleneck.html")


#  fourth plot:  CROWD DENSITY


zones = ["North","NE","East","SE","South","SW","West","NW"]

fig4 = go.Figure()
for key, sc in SCENARIOS.items():
    fig4.add_trace(go.Bar(
        name=sc["label"],
        x=zones,
        y=peak_density[key],
        marker_color=sc["color"],
        text=[f"{v}" for v in peak_density[key]],
        textposition="outside"
    ))

fig4.add_hline(
    y=7, line_dash="dash", line_color="red",
    annotation_text="Critical density (7 people/m²)"
)
fig4.update_layout(
    title="Crowd Density by Stadium Gate (people/m²)",
    xaxis_title="Stadium zone",
    yaxis_title="Density (agents/m²)",
    barmode="group",
    height=450,
    template="plotly_white"
)
fig4.write_html("results/plot4_density.html")
print("Saved plot4_density.html")


#  fifth plot: people evacuating per second


fig5 = go.Figure()
for key, sc in SCENARIOS.items():
    df   = data[key]
    flow = df["evacuated"].diff().fillna(0).rolling(5, center=True, min_periods=1).mean()
    fig5.add_trace(go.Scatter(
        x=df["second"], y=flow,
        name=sc["label"],
        line=dict(color=sc["color"], width=2)
    ))

fig5.update_layout(
    title="People evacuating per second",
    xaxis_title="Time (seconds)", yaxis_title="People per second",
    height=420, template="plotly_white"
)
fig5.write_html("results/plot5_flowrate.html")
print("Saved plot5_flowrate.html")


#  sixth plot: SENSITIVITY ANALYSIS


fig6 = go.Figure()
for label, df, color in zip(sens_labels, sens_data, sens_colors):
    flow = df["evacuated"].diff().fillna(0).rolling(5, center=True, min_periods=1).mean()
    fig6.add_trace(go.Scatter(
        x=df["second"], y=df["evacuated"],
        name=label,
        line=dict(color=color, width=2),
        mode="lines"
    ))

fig6.add_hline(y=5000, line_dash="dot", line_color="gray", annotation_text="Full (5000)")
fig6.add_hline(y=4500, line_dash="dash", line_color="red",  annotation_text="90% cleared")
fig6.update_layout(
    title="Sensitivity Analysis :Effect of Panic Factor on Evacuation",
    xaxis_title="Time (seconds)", yaxis_title="People evacuated",
    height=450, template="plotly_white"
)
fig6.write_html("results/plot6_sensitivity.html")
print("Saved plot6_sensitivity.html")


# SAFETY RECOMMENDATIONS &REFERENCES  
recs_html = """
<ul style="line-height:2; font-size:15px;">

  <li><strong>Add 2 east-side exits</strong> : Bottleneck analysis shows that exits N and NE
  absorb the highest queues across all scenarios. Two additional exits would redistribute the crowd load</li>

  <li><strong>Widen exits E and W</strong>: These exits serve the largest seating sections as seen in stadium standard reports as seen in reports such as Infra solutions (2025)
  Widening reduces the per-person throughput time and directly lowers the crowd density observed at these gates in the panic scenario.</li>

  <li><strong>Reduce capacity to 4,200</strong> : The normal scenario (5,000 agents, 2 blocked
  exits) exceeded the 7 people/m2 critical density (Still, 2000). Capping attendance at 4,200  might bring the worst-case
  milestone time within the safe threshold without infrastructure changes.</li>

  <li><strong>Train exit stewards</strong> : Stewards positioned at high-queue exits can actively
  redirect agents to underused exits, directly reducing the herding effect modelled in the
  panic scenario(Evacuscape, 2024).</li>

  <li><strong>Run evacuation drills twice yearly</strong> : Familiarity with exit locations
  reduces panic and confusion. Drills are zero-cost  and have the highest
  long-term impact on behaviour.</li>


</ul>
"""

recs2_html = """
<ul style="line-height:2; font-size:15px;">

  <li>Evacuscape. (2024, November 28). Enhancing Crowd Safety: Best Practices for Event Evacuation Planning | Evacuscape. Evacuscape. https://evacuscape.com/crowd-safety-how-to-improve-evacuation-planning-for-large-events/</li>

  <li>Infra solutions. (2025, December 13). Stadium seating layout and sightlines. JMR INFRASOLUTIONS. https://jmrinfrasolutions.com/blog/stadium-seating-layout-and-sightlines/</li>     

  <li>Still, G. (2000). Crowd Dynamics. https://www.gkstill.com/Support/Links/Documents/2000_still.pdf</li>                                                     
</ul>
"""


#  BUILD REPORT


print("\nBuilding report.html...")

print("\nLogistic Fit Parameters:")
print(f"{'Scenario':<15} {'K':>8} {'r':>8} {'t0 (s)':>10} {'R²':>8} {'RMSE':>8}")
print("-" * 60)
for key in SCENARIOS:
    f  = fits[key]
    st = fit_stats[key]
    if f is not None:
        K, r, t0 = f
        print(f"{SCENARIOS[key]['label']:<15} {K:>8.0f} {r:>8.4f} {t0:>10.1f} {st['r2']:>8} {st['rmse']:>8}")

print("\nMilestone Times (seconds):")
print(f"{'Scenario':<15} {'25%':>6} {'50%':>6} {'75%':>6} {'90%':>6} {'100%':>6}")
print("-" * 50)
for key in SCENARIOS:
    ms  = milestones[key]
    row = [ms.get(p, "--") for p in [25,50,75,90,100]]
    print(f"{SCENARIOS[key]['label']:<15} " + "  ".join(f"{str(v):>6}" for v in row))

html = """<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Sikaville Stadium - Evacuation Report</title>
  <style>
    body  { background:#F5F0E8; color:#1a1a1a; font-family:futura; padding:24px; max-width:1100px; margin:auto }
    h1    { color:#1B5F8C; border-bottom:2px solid #1B5F8C; padding-bottom:8px }
    h2    { color:#2A7DAF; margin-top:32px }
    table { border-collapse:collapse; width:100%; margin:10px 0 }
    th    { background:#1B5F8C; color:#F5F0E8; padding:6px 12px; text-align:left }
    td    { padding:6px 12px; border-bottom:1px solid #C8C0B0 }
    tr:nth-child(even) { background:#EAE4D8 }
    .warn { color:#C0392B }
    .ok   { color:#1B5F8C }
    iframe{ width:100%; height:520px; border:none; margin:6px 0; border-radius:6px }
  </style>
</head>
<body>
<h1>Sikaville Stadium : Evacuation Risk Assessment</h1>
<p>Group 5</p>

<h2>Milestone Times</h2>
<table>
  <tr><th>Scenario</th><th>25%</th><th>50%</th><th>75%</th><th>90%</th><th>100%</th></tr>
"""
def fmt(v):
        if v is None: return "—"
        cls = "warn" if v > 180 else "ok"
        return f"<span class='{cls}'>{v}s</span>"
for key in SCENARIOS:
    ms = milestones[key]
    html += f"<tr><td>{SCENARIOS[key]['label']}</td>"
    html += "".join(f"<td>{fmt(ms.get(p))}</td>" for p in [25,50,75,90,100])
    html += "</tr>\n"

html += """</table>

<h2>Logistic Fit Parameters</h2>
<table>
  <tr><th>Scenario</th><th>K (capacity)</th><th>r (rate)</th><th>t0 (midpoint)</th><th>R²</th><th>RMSE (hold-out)</th></tr>
"""
for key in SCENARIOS:
    f  = fits[key]
    st = fit_stats[key]
    if f is not None:
        K, r, t0 = f
        html += (f"<tr><td>{SCENARIOS[key]['label']}</td>"
                 f"<td>{K:.0f}</td><td>{r:.4f}</td><td>{t0:.1f}s</td>"
                 f"<td>{st['r2']}</td><td>{st['rmse']}</td></tr>\n")

html += "</table>\n<h2>Charts</h2>\n"
for fname, title in [
    ("plot1_timeline.html",      "Timeline & Logistic Fit Plot"),
    ("plot2_milestones.html",    "Milestone Times"),
    ("plot3_bottleneck.html",    "Bottleneck Analysis"),
    ("plot4_density.html",       "Crowd Density"),
    ("plot5_flowrate.html",      "Flow of People"),
    ("plot6_sensitivity.html",   "Sensitivity Analysis"),
    
]:
    html += f"<h3>{title}</h3><iframe src='{fname}'></iframe>\n"
html += "<h2>Safety Recommendations</h2>\n"
html += recs_html

html += "<h2>References</h2>\n"
html += recs2_html
html += "</body></html>"

with open("results/report2.html", "w") as f:
    f.write(html)