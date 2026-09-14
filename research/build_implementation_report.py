#!/usr/bin/env python3
"""CPU-only implementation report from existing files. Never starts simulation.

Uses only Matplotlib/NumPy/Pillow already installed in the research environment.
Original logs, checkpoints, images and research notes are read-only inputs.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DESKTOP = ROOT.parents[1] / "Desktop"
RUN_NAMES = [
    "20260912_cat_generated_full_shared_gpu_v1",
    "20260912_cat_generated_full_16640_v1",
    "20260912_cat_generated_full_16768_v1",
    "20260913_cat_generated_full_16384_v1",
    "20260914_cat_generated_full_24576_noeval_v1",
    "20260914_cat_generated_full_25344_noeval_v1",
]
NAVY, TEAL, BLUE = "#132D46", "#087F8C", "#3766A3"
ORANGE, RED, GRAY = "#C87720", "#B34443", "#5B6977"
PALE, LINE = "#F0F5F8", "#D5DFE7"
COLORS = [TEAL, BLUE, ORANGE, "#7752A4"]
W, H = 8.27, 11.69
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.compression": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": GRAY, "ytick.color": GRAY,
    "axes.edgecolor": LINE, "axes.grid": True, "grid.color": LINE,
    "grid.alpha": .65, "grid.linewidth": .5,
    "axes.titlesize": 10, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})


class Evidence:
    def __init__(self):
        self.files = {}

    def register(self, path):
        path = Path(path).resolve()
        if str(path) not in self.files:
            self.files[str(path)] = dict(bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return path

    def json(self, path):
        return json.loads(self.register(path).read_text())

    def lines(self, path):
        return [json.loads(s) for s in self.register(path).read_text().splitlines() if s]


def load_data(e):
    runs, allrows, evaluations = [], [], []
    previous_steps = previous_updates = 0
    for name in RUN_NAMES:
        path = RESEARCH / "runs" / name
        config = e.json(path / "config.json")
        rows = e.lines(path / "metrics.jsonl")
        status = e.json(path / "status.json")
        latest = e.json(path / "latest.json")
        initial = sum(config.get("initial_phase_transitions", []))
        if name == RUN_NAMES[0]: initial = 0
        assert initial == previous_steps, (name, initial, previous_steps)
        assert rows[0]["total_steps"] > previous_steps
        assert rows[-1]["total_steps"] == status["total_steps"] == latest["total_steps"]
        assert rows[-1]["optimizer_updates"] > previous_updates
        for row in rows:
            row = dict(row, run_name=name, num_envs=config["args"]["num_envs"])
            allrows.append(row)
        ev = []
        for p in sorted(path.glob("evaluation_*.json")):
            full = e.json(p)
            rec = {k: v for k, v in full.items() if k != "episode_records"}
            rec["file"] = str(p.relative_to(ROOT)); ev.append(rec); evaluations.append(rec)
        runs.append(dict(name=name, path=path, config=config, rows=rows, status=status,
            latest=latest, evaluations=ev, initial=initial,
            median_eps=float(np.median([r["env_steps_per_second"] for r in rows])),
            max_process_gib=max((r["gpu_process_gib"] for r in rows
                                 if "gpu_process_gib" in r), default=None),
            min_free_gib=min(r["cuda_free_gb"] for r in rows)))
        previous_steps = status["total_steps"]; previous_updates = status["optimizer_updates"]
    assert len({r["total_steps"] for r in allrows}) == len(allrows)
    assert runs[-1]["status"]["state"] == "completed"
    final_checkpoint = e.register(runs[-1]["path"] / runs[-1]["latest"]["checkpoint"])
    assert e.files[str(final_checkpoint)]["sha256"] == runs[-1]["latest"]["sha256"]
    generalist = [r for r in allrows if r["phase"] == 9]
    last20 = generalist[-20:]; n = sum(r["episodes"] for r in last20)
    tail = {key: sum(r[key]*r["episodes"] for r in last20)/n
            for key in ("success_rate", "fall_rate", "collision_rate")}
    tail.update(episodes=n, rows=20, start_steps=last20[0]["total_steps"],
                mean_reward=float(np.mean([r["reward"] for r in last20])))
    bank = e.json(RESEARCH / "runs/20260912_cat_generated_bank_v2/bank.json")
    comparison = e.json(RESEARCH / "runs/20260912_cat_direct_comparison/comparison.json")
    baseline_dirs = ["20260911T112807_060354Z_stair_p1", "20260911T133959_984177Z_curb",
        "20260911T134106_554169Z_slope_evaluation", "20260911T134150_217188Z_sitting_evaluation"]
    baseline = [e.json(RESEARCH/"runs"/name/"metrics/metrics_eval.json")["eval/all/mpjpe_g"]
                for name in baseline_dirs]
    for name in ["M0_RESULTS.md", "CAT_DIRECT_EVALUATION.md", "CAT_DISTILLATION.md",
                 "CAT_STYLE.md", "CAT_PARALLEL_TRAINING.md", "PELVIS_GUIDANCE.md",
                 "SCENE_COMPOSITION.md", "ENVIRONMENT_REVIEW.md", "M2_TRAINING.md",
                 "CAT_GRAIL_INTEGRATION.md", "OBSERVATION_SHADOW.md"]:
        e.register(RESEARCH/name)
    for name in ["cat_parallel_train.py", "cat_parallel_env.py", "cat_parallel_core.py",
                 "cat_parallel_policy.py", "cat_distill_model.py", "cat_parallel_bank.py"]:
        e.register(RESEARCH/name)
    proposal = ROOT.parents[1] / "research/research/GRAIL-CAT - Terrain-Aware Whole-Body Control Proposal.md"
    e.register(proposal)
    provenance = e.json(RESEARCH/"provenance.json")
    return dict(runs=runs, rows=allrows, evaluations=evaluations, generalist=generalist,
        tail=tail, bank=bank, comparison=comparison, baseline=baseline, provenance=provenance,
        final_checkpoint=str(final_checkpoint), final_steps=previous_steps,
        final_updates=previous_updates, revision=subprocess.check_output(
            ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())


class Page:
    def __init__(self, report, title, subtitle="", section="IMPLEMENTATION REPORT"):
        self.report = report
        self.number = len(report.pages)+1
        self.fig = plt.figure(figsize=(W,H), dpi=120, facecolor="white")
        self.artists = []
        self.record = [title, subtitle]
        self.text(.58,.34,7.1,section,8.2,TEAL,weight="bold")
        self.text(.58,.76,7.1,title,23,NAVY,weight="bold",leading=1.12)
        if subtitle: self.text(.60,1.27,7.04,subtitle,9.3,GRAY)
        self.fig.add_artist(plt.Line2D([.58/W,7.69/W],[1-1.65/H]*2,color=LINE,lw=.8))
        self.text(.60,11.13,6.4,"GRAIL × CAT  |  Skvayzer research  |  14 September 2026",7.2,GRAY)
        self.text(7.22,11.13,.5,f"{self.number:02d}",8,TEAL,weight="bold")

    def wrap(self,s,width,size,weight="normal",family="DejaVu Sans"):
        renderer=self.fig.canvas.get_renderer()
        prop=FontProperties(family=family,size=size,weight=weight)
        lines=[]
        for raw in str(s).split("\n"):
            line=""
            for word in raw.split():
                trial=(line+" "+word).strip()
                if line and renderer.get_text_width_height_descent(trial,prop,False)[0] > width*self.fig.dpi:
                    lines.append(line); line=word
                else: line=trial
            lines.append(line)
        return "\n".join(lines)

    def text(self,x,y,w,s,size=10.2,color=NAVY,weight="normal",leading=1.32,url=None):
        wrapped=self.wrap(s,w,size,weight)
        a=self.fig.text(x/W,1-y/H,wrapped,fontsize=size,color=color,fontweight=weight,
            va="top",linespacing=leading,url=url)
        self.artists.append(a); self.record.append(str(s))
        return y+(wrapped.count("\n")+1)*size/72*leading

    def para(self,y,s,**kw): return self.text(.60,y,7.05,s,**kw)+.12

    def heading(self,y,s): return self.para(y,s,size=12,weight="bold",color=TEAL)+.04

    def ax(self,x,y,w,h): return self.fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])

    def image(self,path,x,y,w,h):
        path=self.report.evidence.register(path)
        ax=self.ax(x,y,w,h); ax.axis("off")
        with Image.open(path) as im: ax.imshow(np.asarray(im.convert("RGB")))

    def caption(self,y,s): return self.para(y,s,size=8.4,color=GRAY)

    def box(self,y,title,body,color=TEAL):
        title_h=(self.wrap(title,6.7,11,"bold").count("\n")+1)*11/72*1.32
        body_h=(self.wrap(body,6.7,10).count("\n")+1)*10/72*1.32
        h=.28+title_h+.09+body_h
        self.fig.add_artist(Rectangle((.58/W,1-(y+h)/H),7.1/W,h/H,
            transform=self.fig.transFigure,facecolor=PALE,edgecolor="none",zorder=0))
        self.fig.add_artist(Rectangle((.58/W,1-(y+h)/H),.04/W,h/H,
            transform=self.fig.transFigure,facecolor=color,edgecolor="none"))
        self.text(.76,y+.12,6.7,title,11,color,weight="bold")
        self.text(.76,y+.12+title_h+.09,6.7,body,10)
        return y+h+.18

    def table(self,y,headers,rows,widths=None,size=9.0):
        x=.60; total=7.05
        widths=np.asarray(widths if widths else [1]*len(headers),dtype=float)
        widths=widths/widths.sum()*total
        for ri,row in enumerate([headers]+list(rows)):
            weight="bold" if ri==0 else "normal"
            wrapped=[self.wrap(str(s),float(w)-.15,size,weight) for s,w in zip(row,widths)]
            h=max(s.count("\n")+1 for s in wrapped)*size/72*1.28+.16
            self.fig.add_artist(Rectangle((x/W,1-(y+h)/H),total/W,h/H,
                transform=self.fig.transFigure,facecolor=NAVY if ri==0 else (PALE if ri%2 else "white"),
                edgecolor="none",zorder=0))
            xx=x
            for s,w in zip(wrapped,widths):
                self.text(xx+.075,y+.075,float(w)-.15,s,size,"white" if ri==0 else NAVY,weight,1.28)
                xx+=w
            y+=h
        return y+.15

    def source(self,s): self.text(.60,10.72,7.05,"Evidence: "+s,7.6,GRAY)

    def finish(self):
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        for ax in self.fig.axes:
            if not ax.axison:
                continue
            # Tight bounds include only rendered ticks, not unused locator ticks
            # outside the axis limits (or hidden image-axis decorations).
            b = ax.get_tightbbox(renderer)
            if b.x0 < -1 or b.x1 > W*self.fig.dpi+1 or b.y0 < 0:
                raise ValueError(f"Page {self.number} plot outside page")
        for a in self.artists:
            if not a.get_visible() or not a.get_text():
                continue
            b=a.get_window_extent(renderer)
            if b.x0 < -1 or b.x1 > W*self.fig.dpi+1 or b.y0 < 0:
                raise ValueError(f"Page {self.number} text outside page: {a.get_text()[:100]}")
        self.report.pdf.savefig(self.fig)
        self.fig.savefig(self.report.out/f"page_{self.number:02d}.png",dpi=85)
        self.report.pages.append(dict(number=self.number,title=self.record[0],text=self.record))
        plt.close(self.fig)


class Report:
    def __init__(self,out,evidence,data):
        self.out=out;self.evidence=evidence;self.data=data;self.pages=[]
        self.pdf=PdfPages(out/"GRAIL_CAT_Implementation_Report_20260914.pdf",metadata={
            "Title":"GRAIL × CAT — Implementation and Training Report",
            "Author":"Skvayzer research project; prepared with Codex",
            "Subject":"Evidence-based implementation status; no new policy evaluations",
            "Keywords":"GRAIL CAT Unitree G1 whole-body reinforcement learning Isaac Lab"})

    def page(self,*args,**kw): return Page(self,*args,**kw)


def smooth(y,n=7):
    y=np.asarray(y,dtype=float)
    return np.array([y[max(0,i-n+1):i+1].mean() for i in range(len(y))])


def metric_line(ax,rows,key,color,label=None,percent=False,smoothing=7):
    x=np.array([r["total_steps"] for r in rows])/1e6
    y=np.array([r[key] for r in rows])*(100 if percent else 1)
    ax.plot(x,y,color=color,alpha=.23,lw=.7)
    ax.plot(x,smooth(y,smoothing),color=color,lw=1.6,label=label)
    return x,y


def node(ax,xy,wh,title,detail,color=TEAL):
    x,y=xy;w,h=wh
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.016",
        edgecolor=color,facecolor=PALE,lw=1.1))
    ax.text(x+w/2,y+h*.66,title,ha="center",va="center",fontsize=10,fontweight="bold",color=color)
    ax.text(x+w/2,y+h*.30,detail,ha="center",va="center",fontsize=8.1,color=NAVY)


def arrow(ax,a,b,color=GRAY,style="-"):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=10,lw=1.2,color=color,linestyle=style))


def make_report(r):
    d=r.data; rows=d["rows"]; gen=d["generalist"]; tail=d["tail"]
    # 1 — scope and conclusion
    p=r.page("GRAIL × CAT", "Implementation, evidence and training outcomes",section="RESEARCH PROGRESS  /  SEPTEMBER 11–14, 2026")
    p.text(.60,1.94,7,"From a terrain motion prior to obstacle-aware whole-body traversal",20,NAVY,weight="bold",leading=1.15)
    p.para(2.90,"This report traces the proposal into code, simulator diagnostics, generated environments, a CAT-to-GRAIL learning interface and the completed multi-stage training run. Figures and plots use saved artifacts; no new simulator or policy-evaluation job was started.")
    for x,value,label in [( .62,"227.37M","cumulative transitions"),(3.02,"25,344","largest parallel batch"),(5.43,"180 / 45","training / held-out layouts")]:
        p.text(x,4.02,2.2,value,25,TEAL,weight="bold");p.text(x,4.55,2.2,label,9.2,GRAY)
    p.image(DESKTOP/"GRAIL_CAT_review_20260911_v2/01_isaac_frozen_stairs_cat.png",.60,5.00,7.05,3.33)
    p.caption(8.36,"Existing Isaac capture: the frozen GRAIL terrain policy follows a reference with CAT clutter present. This is a foundation/integration demonstration, not the final learned policy or a successful learned avoidance result.")
    p.box(9.10,"Bottom line","The infrastructure and training schedule are implemented and completed. Robust traversal, retained terrain competence of the new policy, manipulation and real-robot deployment are not demonstrated. Final-checkpoint evaluation is intentionally deferred to the user.",RED)
    p.source("Local proposal, pinned code, six linked production runs and existing review media. Snapshot: 14 September 2026.")
    p.finish()

    # 2 — proposal versus implementation
    p=r.page("What was proposed — and built", "The research objective is broader than the completed flat-clutter training stage.")
    y=p.para(1.92,"The proposal combines a terrain-trained GRAIL/SONIC foundation with CAT-style obstacle reasoning, then adds manipulation-compatible commands and a task adapter. GRAIL and CAT provide different interfaces and skills; merging repositories or outputting 29 joints is not evidence that those skills compose. [1–4]")
    y=p.table(y,["Milestone","Implemented evidence","Current limit"],[
        ["M0 · terrain foundation","Pinned installation; strict terrain checkpoint restoration; stairs/curb/slope/sitting sample rollouts.","Small released-data samples, not generalization or retention of our trained policy."],
        ["M1 · geometry / contact","CAT mesh/field port, support separation, collider covers, placement screens, guidance and physical diagnostics.","Geometry checks do not certify balance or dynamic whole-body passage."],
        ["M2 · terrain + clutter","Reference-conditioned residual machinery and reviewed arm-clearance scenes; bounded pilot attempted.","Early pilot failed. This is not the later generated-clutter learner."],
        ["M3 · goal-directed student","Frozen CAT features + trainable latent adapter + frozen GRAIL decoder; generated flat-clutter transfer/DAgger/PPO.","Training completed, but logged traversal performance is poor; no validated stair/clutter student."],
        ["M4/M5 · manipulation","Articulated geometry and kinematic posture witnesses support future work.","No masked hand-target policy, load curriculum, grasp/carry/place adapter or FABRICS implementation."],
        ["M6 · perception / deployment","Separate earlier CAT perception-preview work exists in humanoid_navigation.","This learner still uses oracle fields; no sensor-trained student or learned-policy robot deployment."],
    ],[1.2,2.95,2.9],8.8)
    y=p.box(y+.06,"Two implementations must not be conflated","A: the reference-conditioned GRAIL terrain/clutter diagnostic stack. B: the later CAT-style generated flat-floor task with a GRAIL decoder. The completed large training run is B, not a full realization of A plus manipulation.")
    p.source("Obsidian terrain-aware proposal; CAT_GRAIL_INTEGRATION.md; M2_TRAINING.md; CAT_PARALLEL_TRAINING.md.")
    p.finish()

    # 3 — foundation
    p=r.page("Reproducing the terrain foundation", "Preserve a known starting point before adding obstacle-aware learning.")
    y=p.para(1.92,"An isolated GRAIL-CAT worktree and environment were created around the released terrain model. Configuration, weights, dataset and simulator revisions were pinned; original artifacts remained immutable. Restoration resolves actual joint names, action scales, observation history and the motor-token interface rather than guessing from a robot configuration label.")
    ax=p.ax(.95,3.02,6.45,2.35)
    bars=ax.bar(["Stairs","Curb","Slope","Sitting"],d["baseline"],color=[TEAL,BLUE,ORANGE,"#7752A4"],width=.56)
    ax.set_ylabel("Global MPJPE (mm)");ax.set_ylim(0,65);ax.bar_label(bars,fmt="%.2f",padding=4,fontsize=9)
    p.caption(5.65,"Figure 1. Existing unmodified-GRAIL reproduction results: one alphabetically selected released scene/motion per family, seed 42. Stairs/curb/slope: 499 control steps; sitting: 248. No failure termination was recorded. Lower MPJPE indicates better reference tracking, not goal-directed traversal success.")
    y=p.heading(6.50,"Resolved foundation contract")
    y=p.table(y,["Item","Implementation"],[
        ["Embodiment","29 joints: 12 legs + 3 waist + 7 per arm; finger/gripper control is separate."],
        ["Runtime","Isaac Sim 5.1; Isaac Lab 2.3.2; Python 3.11; PyTorch 2.7.0 + CUDA 12.8; RTX 5090."],
        ["Terrain checkpoint","terrain_release/last.pt, not a claim that a manipulation adapter is interchangeable."],
        ["Baseline dynamics","Released 50 Hz control / 200 Hz physics. Later native-CAT task uses 500 Hz physics."],
        ["Backward path","Historical 4-env, 2-update PPO smoke saved finite weights/optimizer state. It tested plumbing, not improved behavior."],
    ],[1.65,5.4],9)
    p.source("M0_RESULTS.md and four original metrics/metrics_eval.json files; research/provenance.json.")
    p.finish()

    # 4 — geometric interfaces
    p=r.page("Geometry: support is not free space", "Three semantics, not a single height map or one global collision exemption.")
    ax=p.ax(.65,1.97,7,2.50);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    node(ax,(.015,.71),(.97,.22),"Shared, explicitly transformed physical scene","Terrain mesh + CAT mesh + articulated robot collision geometry",NAVY)
    for x,title,detail,col in [( .015,"Support surfaces","Height / normal / validity\nCandidate footholds",TEAL),(.352,"Forbidden clutter","Signed mesh distance\nBody clearance",RED),(.689,"Contact permissions","Link / surface / phase\nAllowed support contact",BLUE)]:
        node(ax,(x,.16),(.294,.34),title,detail,col);arrow(ax,(x+.147,.70),(x+.147,.51))
    p.caption(4.60,"Figure 2. Implemented geometry design in the terrain diagnostic stack. Open terrain surfaces use unsigned distance and explicit support information; they are not assigned an invented inside/outside sign. Permission for foot–tread contact does not permit shin–riser or hand–rail collision.")
    y=p.para(5.43,"CAT meshes and fields share one XYZ/yaw transform. Exact closed-mesh signed distance provides an independent check; 104 conservative spheres cover imported robot collision primitives. A negative cover gap flags a reference for inspection, not proof of exact visual-mesh collision or universal task infeasibility.")
    y=p.para(y,"Full-height volumes preserve all reference probe bounds. In the documented stair composition, field coverage increased from 74.46% to 100% of 51,896 saved probe samples; the fitted 4 cm grid was 56 × 74 × 72 (2.88 m high). Outside-grid queries remain invalid. These are simulation-oracle fields, not sensor visibility maps.")
    ax=p.ax(1.50,7.65,2.5,1.52)
    b=ax.barh(["Nearest\nsupport cell","Checked\ntransit root"],[334,499],color=[ORANGE,TEAL]);ax.set_xlim(0,620)
    ax.bar_label(b,labels=["334 / 499","499 / 499"],padding=4,fontsize=8);ax.set_xlabel("Valid guidance frames");ax.invert_yaxis()
    p.text(4.60,7.54,3.02,"Pelvis transit fix",11,TEAL,weight="bold")
    p.text(4.60,7.93,3.02,"The pelvis can cross a tread edge without standing on it. A bounded, checked connector joins the actual pelvis root to reachable support nodes: ≤0.28 m horizontally and ≤0.20 m height excursion. All crossed cells must pass the transit screen.",9.1)
    p.caption(9.68,"Figure 3. Existing replay recovered 165 rejected frames without losing previously valid frames. The graph remained unchanged; independent connector witnesses passed. This is geometric validity, not a balance or continuous-control guarantee.")
    p.source("SCENE_COMPOSITION.md; PELVIS_GUIDANCE.md; cat geometry, support graph and observation-shadow modules.")
    p.finish()

    # 5 — physical scene versus kinematic witness
    p=r.page("Composing clutter with stairs", "Orientation, support, passage and reference conflict are checked separately.")
    y=p.para(1.92,"The original CAT generator was retained, with traceable floor/lateral/overhead roles. Terrain placement uses explicit XY/yaw and, when requested, a rigid Z translation grounded on a nearly level support footprint. It does not warp obstacles over stairs, bury hurdles, invent supports or relabel a blocked reference as successful avoidance.")
    p.image(DESKTOP/"GRAIL_CAT_environment_review_20260912_v2/02_arm_clearance_closeup_conflict.png",.60,3.13,7.05,3.55)
    p.caption(6.79,"Figure 4. Existing offline kinematic review: a collision-screened arm posture changes the arms while preserving root, waist, legs and the stair reference. This is not learned avoidance and does not show that the controller can execute the posture while balancing. Geometry-validation seed 102 is shown, not a final unseen-policy test.")
    y=p.table(7.67,["Documented placement / witness","Result and interpretation"],[
        ["Original hurdle placement","Reference screen passed, but hurdle was buried: rejected as a retained obstacle challenge."],
        ["Grounded landing hurdle","Roles retained; 70 reference frames flagged, minimum cover gap −0.043 m. A changed motion is needed."],
        ["Clear obstacle beyond the endpoint","Passed geometry checks; explicitly a nonblocking control, not hurdle-crossing success."],
        ["Arm witness, case 17 / seed 102","CAT gap 0.03652 m; nonlocal self gap 0.06144 m; feet unchanged. Dynamic feasibility remains unverified."],
    ],[2.55,4.5],8.7)
    p.source("SCENE_COMPOSITION.md; ENVIRONMENT_REVIEW.md; existing review-image provenance.json and posture-witness reports.")
    p.finish()

    # 6 — direct teacher parity
    p=r.page("Checking the CAT simulator transfer", "Released CAT weights directly controlled the native G1 model in both simulators.")
    y=p.para(1.92,"Before teaching GRAIL, the frozen released CAT generalist was run on three native flat fixtures in MuJoCo and Isaac/PhysX. These are prior recordings, reused for this report. CAT still controlled only the 12 legs; the other 17 joints followed nominal PD targets. No GRAIL learner was active in this comparison.")
    cases=d["comparison"]["cases"];x=np.arange(3)
    ax=p.ax(.95,3.12,6.45,2.25)
    for shift,engine,color in [(-.18,"mujoco",TEAL),(.18,"isaac",BLUE)]:
        vals=[c["episodes"][engine]["minimum_native_site_sdf_m"]*1000 for c in cases]
        b=ax.bar(x+shift,vals,.34,label=engine.title(),color=color);ax.bar_label(b,fmt="%.2f",padding=3,fontsize=8)
    ax.axhline(-40,color=RED,ls="--",lw=1,label="Native comparison threshold: −40 mm")
    ax.set_xticks(x,["side1","hurdle1","crouch1"]);ax.set_ylim(-65,200);ax.set_ylabel("Minimum monitored-site SDF (mm)")
    ax.legend(fontsize=7.8,ncol=2,loc="upper left")
    p.caption(5.70,"Figure 5. All six recorded episodes remained upright and ended within 0.2 m XY of the native goal. Isaac overhead clearance crossed the native threshold once by about 0.35 mm: a partial pass, not rounded into success. Episode stopping times and trajectories differ across engines.")
    y=p.table(6.57,["Independent check","Recorded maximum discrepancy"],[
        ["Native OBJ → Isaac world vertices","5.73 × 10⁻⁸ m; triangle indices identical"],
        ["Field sampling on saved trajectories","2.50 × 10⁻⁶ in field values"],
        ["Imported initial robot forward kinematics","7.19 × 10⁻⁷ m"],
        ["Frozen PyTorch actor → original ONNX","6.26 × 10⁻⁷ action units over 1,587 observations"],
    ],[3.55,3.5],9)
    p.box(y+.08,"Compatibility is not identical physics","The teacher uses native legacy interpolation/clamping conventions. Corrected geometric safety queries are separate. Solver/contact differences remain, and these three fixtures do not validate the full generated distribution or the stricter later training success criterion.",ORANGE)
    p.source("CAT_DIRECT_EVALUATION.md; runs/20260912_cat_direct_comparison/comparison.json and onnx_parity.json.")
    p.finish()

    # 7 — diversity
    p=r.page("Generated training environments", "Many parallel robots; a finite, explicitly versioned geometry bank.")
    p.para(1.92,"The unchanged pinned CAT random obstacle and potential-field generators produce lateral, low, overhead and mixed layouts. Native 4 cm fields use XYZ order: 75 × 50 × 38 cells, origin (−0.5, −1, 0), goal (2, 0, 0.75). Each reset resamples a cached scene; it does not generate new geometry online.")
    families=[("lateral",310018), ("low",320018), ("overhead",330018), ("mixed",340018)]
    for i,(family,seed) in enumerate(families):
        p.image(DESKTOP/f"CAT_training_layouts_20260912/{family}_{seed}.png",
                .60+(i%2)*3.57,3.07+(i//2)*2.03,3.48,1.96)
    p.caption(7.18,"Figure 6. Existing generated-bank previews at difficulty 0.6, seed offset 18 in each family. They show geometry, not robot trajectories. Floating forbidden volumes are native generator outputs; the rendering does not imply physical robot–clutter contacts.")
    counts=collections.Counter((s["family"],s["split"]) for s in d["bank"]["scenes"])
    y=p.table(7.95,["Family","Training","Held out"],[[f.title(),counts[f,"train"],counts[f,"validation"]] for f,_ in families]+[["Total",180,45]],[3.3,1.8,1.95],9)
    p.para(y,"240 recipes were requested; 15 low-obstacle recipes became empty/full after native morphology and were rejected without easier-seed retries. Difficulty spans 0.2/0.4/0.6/0.8. The 225 admitted geometries are unique; 25,344 environments are not 25,344 unique layouts.",size=9.1)
    p.source("runs/20260912_cat_generated_bank_v2/bank.json; CAT_TRAINING_LAYOUT_VIDEOS.md and preview manifest.")
    p.finish()

    # 8 — actual network
    p=r.page("The implemented whole-body policy", "A trainable motor-token adapter, not averaging incompatible CAT and GRAIL actions.")
    ax=p.ax(.65,1.95,7,3.70);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    node(ax,(.02,.77),(.43,.18),"CAT observation → frozen trunk","162D native input → 64D features",BLUE)
    node(ax,(.56,.77),(.42,.18),"State / context","1029D history + 64D base + mode",NAVY)
    node(ax,(.23,.44),(.54,.22),"Trainable adapter + Gaussian","1158 → 512 → 256 → 128 → 64 → 64",TEAL)
    node(ax,(.23,.10),(.54,.21),"Frozen GRAIL g1_dyn decoder","64D motor tokens + 1029D state → 29 joints",BLUE)
    arrow(ax,(.23,.76),(.36,.67));arrow(ax,(.77,.76),(.64,.67));arrow(ax,(.50,.43),(.50,.32))
    ax.text(.81,.43,"Separate critic",fontsize=9,fontweight="bold",color=ORANGE)
    ax.text(.81,.34,"1029 + 250D\n→ scalar value",fontsize=8.1,color=ORANGE)
    p.caption(5.76,"Figure 7. Solid pathway is the implemented actor. Frozen components preserve pretrained features/decoder parameters; only the adapter, action log-standard-deviation and separate privileged critic are optimized. The frozen decoder still receives gradients with respect to its token input.")
    y=p.table(6.62,["Tensor / control step","Exact implemented meaning"],[
        ["Adapter input: 1,158D","64 CAT features + normalized 1,029D GRAIL packet + 64 base tokens + one mode bit."],
        ["Action distribution","64D diagonal Gaussian before tanh; log std clamped to [−4, −0.5]."],
        ["Decoder token","base + 2·tanh(z). Continuous post-quantization correction, not editing discrete code indices."],
        ["Executed joint target","Decode [token, state], clip wrapper action, apply checkpoint scale/offset, reorder by joint name."],
        ["Critic: 1,279D input","1,029D normalized GRAIL state + CAT 250D privileged packet; 1024/512/256/128 hidden widths."],
    ],[2.25,4.8],8.8)
    p.para(y,"Flat navigation uses zero base tokens and a static virtual object context, not a future motion clip. Retention replay uses original recorded GRAIL tokens/actions. This is not yet the proposed masked hand-command or live-sensor interface.",size=9)
    p.source("cat_distill_model.py; cat_parallel_policy.py; cat_parallel_train.py; pinned GRAIL action contract.")
    p.finish()

    # 9 — staged learning
    p=r.page("CAT-style staged learning", "Four specialist families, then distillation and generalist PPO.")
    ax=p.ax(.98,2.02,6.3,2.25)
    labels=["Lateral","Low","Overhead","Mixed","Generalist"]
    transfer=np.array([4.194304]*4+[0]);ppo=np.array([16.777216]*4+[134.217728]);dag=np.array([0]*4+[8.388608])
    y=np.arange(5)
    ax.barh(y,transfer,color=TEAL,label="Teacher transfer")
    ax.barh(y,dag,left=transfer,color=ORANGE,label="Specialist DAgger")
    ax.barh(y,ppo,left=transfer+dag,color=BLUE,label="PPO")
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel("Planned transition budget (millions)")
    ax.legend(loc="center right",fontsize=8)
    p.caption(4.87,"Figure 8. Planned total: 226,492,416 transitions. Actual total: 227,373,056; full vector batches overshot two stage targets by 880,640 transitions (0.39%). Completed stages are not replayed when environment count changes.")
    y=p.heading(5.56,"Which loss is actually optimized?")
    y=p.table(y,["Stage","Actor objective","Common terms"],[
        ["Specialist transfer","Absolute CAT leg-target MSE + 0.25 × nominal upper-posture MSE.","Critic value loss + recorded GRAIL all-joint retention MSE."],
        ["Generalist DAgger","KL(whole-body specialist Gaussian || generalist Gaussian), selected by scene family on generalist-visited states.","Critic value loss + retention MSE."],
        ["Specialist / generalist PPO","Clipped likelihood-ratio objective + entropy bonus; stochastic 64D token samples.","Critic value loss + retention MSE. Logged leg MSE is diagnostic, not a PPO imitation term."],
    ],[1.65,3.25,2.15],9)
    y=p.para(y+.03,"The released CAT generalist supplies leg labels, not fabricated arm labels. Teacher assistance is Bernoulli per environment and decays to zero over the first 75% of each transfer budget. Whole-body specialists, not CAT's 12D action head, later teach the generalist token distribution.",size=9.7)
    p.para(y,"PPO preserves the sampled pre-tanh variable and its likelihood; unchanged-policy likelihood is checked before optimization. Terminated versus timed-out episodes use explicit final-state handling, and histories reset per environment.",size=9.7)
    p.source("cat_parallel_train.py; cat_parallel_policy.py; native released policy_config; per-run phase counters.")
    p.finish()

    # 10 — dynamics and divergences
    p=r.page("Task mechanics and fidelity limits", "Borrowed CAT ingredients are distinguished from whole-body adaptations.")
    y=p.table(1.95,["Component","Implemented training setting"],[
        ["Physics / control","Independent GPU PhysX G1 articulations, 5 m spacing; 500 Hz torque PD, 50 Hz policy; 1,000-step episodes."],
        ["Reset distribution","XY offsets ±1 m, yaw ±π/2, joint multiplier 0.5–1.5, initial velocity ±0.5; per-environment histories and scene IDs."],
        ["Randomization","PD gain factors 0.75–1.25, torque perturbations, 1.3–1.5 Hz gait, observation noise, delayed odometry and pushes."],
        ["Scene sampling","Adaptive failure sampling with 0.95 success EMA and nonzero coverage floor; train/holdout split explicit."],
        ["PPO recipe","LR 3e−4; entropy 0.003; γ 0.98; GAE λ 0.95; clip 0.2; 32-step rollout; four epochs × 64 minibatches; gradient norm 1."],
        ["Rewards","22 ported native terms: guidance/clearance, root/orientation tracking, gait/contact, slip/balance, smoothness, limits and torque. Total nonnegative clipping retained."],
    ],[1.6,5.45],9)
    y=p.box(y+.08,"An intentional task difference matters","The current learner runs on a physical flat floor with CAT SDF-only clutter; ordinary robot–clutter physical contacts are not the learning mechanism. This is not the proposal's full physical terrain-and-clutter task. Oracle fields, virtual object features and incomplete pair-specific self-contact termination remain limitations.",ORANGE)
    y=p.heading(y,"Success and collision criteria")
    y=p.para(y,"In this implementation, any of 11 sites with SDF < 0 flags a violation; after a 50-step grace period, violations terminate the episode. Falling uses head height <0.7 m or an inverted pelvis up direction. Success requires the 1,000-step timeout, at least 50 consecutive final steps within 0.2 m of the goal, and no counted earlier violation.",size=9.5)
    y=p.para(y,"These tests are stricter/different from the earlier direct-CAT comparison's −0.04 m monitored-site tolerance and exit-plane stopping. The two success numbers are not interchangeable. Sparse sites also do not certify complete articulated or payload clearance.",size=9.5)
    p.para(y,"Our bank uses procedural layouts only, not CAT's full hybrid indoor-crop distribution. The pinned upstream generalist config specifies 5 billion timesteps and 65,536 environments; our 227M-transition run is a smaller adapted experiment, not full-budget reproduction. [2,4]",size=9.5)
    p.source("cat_parallel_core.py; cat_parallel_env.py; native_recipe in the saved config; CAT_STYLE.md.")
    p.finish()

    # 11 — earlier failures
    p=r.page("Development experiments and pivots", "What changed when early approaches failed.")
    y=p.para(1.93,"The first terrain-clutter M2 learner was a different reference-conditioned residual pilot: four replicas, a 0.1·tanh token residual, one development layout and one geometry-validation layout. The overnight attempt failed at iteration 135 after about six minutes. It is not counted in the later production-run lineage or presented as a completed CAT-style experiment.")
    y=p.para(y,"The project then moved to frozen CAT teaching the frozen GRAIL decoder on flat clutter. Small supervised and DAgger pilots established gradients and checkpoint continuation, but closed-loop tests showed why imitation error and early exit-plane crossing are insufficient.")
    ax=p.ax(1.00,4.01,6.20,2.35)
    values=[.01084,.004733,.003526,.003189]
    b=ax.bar(["200 updates\nstrict pilot","600\ncontact-aware","1,400\nDAgger round 1","2,200\nDAgger round 2"],values,color=[BLUE,TEAL,ORANGE,"#7752A4"])
    ax.set_ylabel("Historical held-out leg MSE (rad²)");ax.set_ylim(0,.013)
    ax.bar_label(b,labels=[f"{v:.4f}" for v in values],padding=4,fontsize=8)
    p.caption(6.87,"Figure 9. Previously recorded small-pilot results, not the generated-bank run. Datasets changed between pilots; these bars are a development history, not a controlled scaling comparison. The 600→1,400→2,200 continuation retained the same held-out rows.")
    y=p.table(7.69,["Observed result","Implication"],[
        ["Lower leg imitation MSE","Shows a supervised objective improved, not that the student balances or completes goals."],
        ["Short rollout crossed x = 1.9","Later full-horizon checks revealed falls and clearance violations; early stopping hid failure."],
        ["One geometry with several starts","Insufficient diversity; superseded by the 180-layout generated training bank."],
        ["Finite gradients / checkpoint parity","Useful implementation evidence, but not a reason to label a policy deployable."],
    ],[2.85,4.2],9)
    p.source("M2_TRAINING.md; CAT_DISTILLATION.md; existing small-pilot summary and saved full-horizon checks.")
    p.finish()

    # 12 — per-specialist learning
    p=r.page("Specialist learning curves", "Actual production lineage; objectives and policies change at stage boundaries.")
    families=["Lateral","Low","Overhead","Mixed"]
    for f in range(4):
        ax=p.ax(.99+(f%2)*3.52,2.25+(f//2)*2.77,2.93,1.93)
        rr=[row for row in rows if row["phase"] in (2*f,2*f+1)]
        for kind,color in [("transfer",TEAL),("ppo",BLUE)]:
            part=[row for row in rr if row["kind"]==kind]
            ax.plot([row["total_steps"]/1e6 for row in part],
                    [row["leg_mse"] for row in part],color=color,lw=1.5,label=kind.upper())
        ax.set_yscale("log");ax.set_title(families[f],loc="left",fontweight="bold")
        ax.set_xlabel("Cumulative transitions (M)");ax.set_ylabel("Leg MSE (rad²)")
        if f==0: ax.legend(fontsize=7.6,loc="lower right")
    p.caption(7.90,"Figure 10. Raw leg-target MSE per logged rollout update. There is no smoothing in these four panels. Transfer and PPO are separate segments: leg MSE is optimized during transfer but only logged during PPO, so a PPO increase is not a contradiction in the implemented loss.")
    y=p.box(8.72,"What these curves do — and do not — establish","The learning pipeline updated parameters across all four families. They do not establish robust traversal or preserved stairs. Transfer is partly teacher-assisted; PPO changes the objective. Batch sizes and shared-GPU conditions also changed, so this is one experimental history rather than a matched ablation.",ORANGE)
    p.para(y,"All charts join only the six actual parent-linked production runs. Capacity smokes, interrupted startups and small-pilot runs are excluded from the cumulative counters and learning curves.",size=9.4)
    p.source("Six metrics.jsonl files joined by verified initial/final transition counters; source run list on page 16.")
    p.finish()

    # 13 — main outcome
    p=r.page("Generalist: completed, not solved", "Training metrics from 92.40M to 227.37M cumulative transitions.")
    for i,(key,title,color) in enumerate([
        ("reward","Mean per-step reward",TEAL),("fall_rate","Completed-episode fall rate (%)",RED),
        ("collision_rate","Completed-episode collision indicator (%)",ORANGE),("retention_mse","Recorded action-retention MSE (rad²)",BLUE)]):
        ax=p.ax(.99+(i%2)*3.52,2.23+(i//2)*2.76,2.94,1.90)
        metric_line(ax,gen,key,color,percent=key.endswith("rate"))
        ax.set_title(title,fontsize=8.8,loc="left");ax.set_xlabel("Cumulative transitions (M)")
        ax.axvline(101.838848,color=GRAY,ls=":",lw=.8)
        if key.endswith("rate"): ax.set_ylim(0,105)
    p.caption(7.85,"Figure 11. Thin lines are raw metrics; thick lines are trailing seven-update means. Dotted boundary: policy evaluation disabled after 101.84M transitions. Rates are logged training episode indicators, not held-out success; fall and collision can overlap. Rewards are per-step means, not episode returns.")
    p.box(8.63,"Final 20 logged updates: zero recorded successes",
        f"Across {tail['episodes']:,} completed training episodes: success {tail['success_rate']*100:.1f}%, fall {tail['fall_rate']*100:.1f}%, collision indicator {tail['collision_rate']*100:.1f}% (episode-weighted). Mean per-step reward was {tail['mean_reward']:.5f}. This is a serious training-performance limitation, not a final deterministic evaluation.",RED)
    p.source("Final generalist metrics, last 20 rollout records; no new evaluation, inference or physics was run for this report.")
    p.finish()

    # 14 — recorded eval history
    p=r.page("What the existing evaluations show", "Earlier evaluations only; the final checkpoint remains unevaluated.")
    y=p.para(1.94,"Eight full-horizon evaluation files were produced before the user's no-evaluation request, plus one interrupted early evaluation. They are reported here rather than rerun. Each full horizon spans 1,000 control steps with automatic resets; its many completed episodes are not independent training seeds or complete trajectories of equal length.")
    ev=[x for x in d["evaluations"] if x.get("full_horizon")]
    phase_ends={row["total_steps"]:row for row in rows}
    table=[]
    for item in ev:
        row=phase_ends[item["total_steps"]]
        table.append([f"{item['total_steps']/1e6:.2f}",row["family"].title()+" / "+row["kind"],
            f"{100*item['success_rate']:.3f}",f"{100*item['fall_rate']:.1f}",f"{100*item['collision_rate']:.1f}"])
    y=p.table(3.25,["Steps (M)","Evaluated stage","Success %","Fall %","Collision %"],table,[1,2.65,1.15,1.05,1.2],8.9)
    y=p.para(y+.12,"The earlier 4.19M-transition lateral evaluation was interrupted at 768 steps and is excluded from this full-horizon table. Small nonzero transfer-stage success rates did not persist through the recorded specialist PPO endpoints. Generalist-after-DAgger evaluation was also unsuccessful.",size=9.7)
    y=p.box(y+.02,"No hidden evaluation after the requested stop","Both no-evaluation continuations contain zero evaluation_*.json files. Their config and logged metrics say evaluations_enabled=false. The inherited unique_validation_scenes_visited=45 is cumulative checkpoint state, not evidence that new validation episodes ran.")
    p.para(y,"The final 227.37M checkpoint is available for the user's later evaluation. A credible performance claim still needs fixed held-out starts/geometries, unassisted full-horizon rollout metrics, terrain-retention checks and independent training seeds. Nothing in this report certifies real-robot safety.",size=9.8)
    p.source("Existing evaluation_*.json and config/status files only. Final W&B: wandb.ai/skvayzer/grail-cat/runs/rmx6awvl.")
    p.finish()

    # 15 — resource use
    p=r.page("GPU scaling and time accounting", "More memory increased batch size; it did not demonstrate better learning.")
    ax=p.ax(.95,2.17,6.45,2.02)
    for i,run in enumerate(d["runs"]):
        if i==0: continue
        xs=[x["total_steps"]/1e6 for x in run["rows"]];ys=[x["gpu_process_gib"] for x in run["rows"]]
        ax.plot(xs,ys,lw=1.4,label=f"{run['config']['args']['num_envs']:,}")
    ax.set_xlabel("Cumulative transitions (M)");ax.set_ylabel("Total process VRAM (GiB)");ax.set_ylim(14,32)
    ax.legend(ncol=3,fontsize=7.5,title="Environment count",title_fontsize=8,loc="lower right")
    ax=p.ax(.95,5.02,6.45,2.02)
    vals=[run["median_eps"]/1000 for run in d["runs"]]
    b=ax.bar(np.arange(6),vals,color=[GRAY,GRAY,GRAY,TEAL,BLUE,BLUE]);ax.bar_label(b,fmt="%.1f",padding=3,fontsize=8)
    ax.set_xticks(np.arange(6),["2,048","16,640","16,768","16,384","24,576","25,344"])
    ax.set_xlabel("Parallel environments, in chronological run order");ax.set_ylabel("Median training ksteps/s");ax.set_ylim(0,62)
    p.caption(7.48,"Figure 12. NVIDIA process memory includes simulator allocations, unlike Torch-only metrics. Throughput bars are medians of logged training updates; the first three runs shared a heavily occupied GPU, later ones did not. These are not controlled scaling benchmarks. Evaluation/startup/restart pauses are excluded from the per-update rates.")
    last=d["runs"][-1]
    y=p.table(8.38,["Final continuation","Measured / recorded"],[
        ["Maximum logged process VRAM",f"{last['max_process_gib']:.3f} GiB ≈ {last['max_process_gib']*2**30/1e9:.2f} decimal GB"],
        ["Minimum logged CUDA-free memory",f"{last['min_free_gib']:.3f} GiB; stop guard remains 1 GiB"],
        ["Final segment elapsed time",f"{last['rows'][-1]['elapsed_s']/60:.2f} min after learner initialization; no policy evaluations"],
        ["Completion vs target","227.37M actual / 226.49M planned; last whole batches account for the excess."],
    ],[3.3,3.75],9)
    p.source("Per-run metrics.jsonl, config.json and completed status.json. GPU sharing prevents causal speedup claims.")
    p.finish()

    # 16 — reproducibility and implementation map
    p=r.page("Reproducibility and code ownership", "One preserved learner lineage; simulator episodes restart fresh after resize.")
    y=p.table(1.94,["Run suffix / environments","Saved cumulative transitions","Role"],[
        ["shared_gpu_v1 / 2,048","4,194,304","Lateral transfer"],
        ["16640_v1 / 16,640","6,856,704","Lateral PPO continuation"],
        ["16768_v1 / 16,768","10,612,736","Checkpointed memory-guard stop"],
        ["20260913…16384_v1","101,838,848","Specialists, DAgger, early generalist"],
        ["24576_noeval_v1","107,343,872","Generalist PPO, no evaluation"],
        ["25344_noeval_v1","227,373,056","Generalist PPO completed"],
    ],[2.75,2,2.3],8.8)
    y=p.para(y+.03,"Saved state contains all specialist/generalist adapter and critic parameters, action-distribution parameters, normalization, optimizer moments, RNG and per-stage transition counters. Frozen releases are hash-addressed. Resizing preserves transition budgets; it does not reinterpret an old rollout index at the new batch size. It is learner continuation, not bit-exact PhysX replay.",size=9.5)
    y=p.table(y,["Module","Responsibility"],[
        ["cat_parallel_bank / core","Pinned generated bank, field sampling, histories and reward equations."],
        ["cat_parallel_env","Native G1 import, vector physics, randomized resets, targets, final observations and terminations."],
        ["cat_distill_model / parallel_policy","Frozen CAT/GRAIL components, trainable token adapter, critic and learner serialization."],
        ["cat_parallel_train","Transfer/DAgger/PPO, checkpointed resume, W&B, resource checks and --no-eval."],
        ["Geometry / shadow / residual modules","Separate terrain-composition, contact, guidance and earlier reference-conditioned pilot tooling."],
    ],[2.65,4.4],8.7)
    y=p.para(y+.03,"Code: Skvayzer/GRAIL, branch research/grail-cat-terrain. Final training code revision: a8f359f. Report source snapshot: "+d["revision"][:12]+". Training does not connect to ROS/SDK or publish robot commands. Other users' jobs were not killed; one existing deployment PID was explicitly approved for GPU coexistence, while new deployments remained guarded.",size=9.2)
    p.text(.60,y,7.05,"Final checkpoint: research/runs/20260914_cat_generated_full_25344_noeval_v1/\ncheckpoint_000227373056.pt",8.5,TEAL)
    p.source("Parent configs, latest.json/checkpoint hash, source manifests and implementation files. Exact artifact inventory accompanies the PDF.")
    p.finish()

    # 17 — limits and research next steps
    p=r.page("What remains for the proposal", "Engineering progress is real; the intended research claims are still conditional.")
    y=p.box(1.95,"Do not promote this checkpoint to hardware","The final logged training behavior is poor and final evaluation is deferred. A frozen terrain decoder plus 29-joint outputs does not establish terrain retention, obstacle avoidance, useful arm control or manipulation.",RED)
    y=p.heading(y,"Next experiments — proposals, not actions taken")
    y=p.table(y,["Priority","Question / experiment","Evidence required"],[
        ["1 · Diagnose flat failure","Compare frozen CAT and the student under the same reset, full-horizon and SDF criteria. Check the stricter zero-SDF training termination, teacher compatibility and early falls.","Matched rollouts and failure traces; no assumption that more VRAM or more steps resolves it."],
        ["2 · Learn a stable student","Isolate transfer, token capacity, exploration, reward/termination and retention tradeoffs. Use multiple seeds and controlled ablations.","Unassisted held-out goal completion with acceptable falls and clearances."],
        ["3 · Rejoin terrain stack","Integrate support/contact-aware physical clutter and compatible terrain controls. Rehearse and measure original skills.","Stairs, slopes and curbs retained while clutter avoidance improves."],
        ["4 · Manipulation compatibility","Add masked hand/torso targets, mobility permissions, payload geometry and contact tasks; then consider a frozen-base task adapter.","Balance + hand-task success, later grasp/carry/place, not merely nominal arm movement."],
        ["5 · Sensor / robot readiness","Replace oracle actor fields with the intended LiDAR/depth pipeline, uncertainty and timing; export the full history/preprocessing system.","Independent simulation and non-actuating shadow tests before separately approved physical deployment."],
    ],[1.3,3.7,2.05],8.65)
    y=p.heading(y+.04,"Candidate contributions, contingent on experiments")
    p.para(y,"Contact-conditioned support/clutter reasoning; retention-aware conversion of a terrain prior into goal-directed whole-body traversal; manipulation-compatible mobility; and compositional terrain–clutter–task evaluation. These require mechanisms, ablations and evidence. Combining GRAIL and CAT is not itself a novelty claim, and FABRICS remains optional, unimplemented work.",size=9.8)
    p.source("Original terrain-aware proposal; current source and logged failure evidence. No new training or evaluation was launched for this report.")
    p.finish()

    # 18 — references and provenance
    p=r.page("References and evidence index", "Primary sources, exact local artifacts and plotting conventions.")
    refs=[
        ("[1] Xie et al. GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors. arXiv:2606.05160 (2026).","https://arxiv.org/abs/2606.05160"),
        ("[2] Xue et al. Collision-Free Humanoid Traversal in Cluttered Indoor Scenes. arXiv:2601.16035 (2026).","https://arxiv.org/abs/2601.16035"),
        ("[3] NVIDIA GRAIL official code. Local upstream pin: aa31d8242ac79b11545b9e3635f73014a227bdfc.","https://github.com/NVlabs/GRAIL"),
        ("[4] Click-and-Traverse official code. Generator pin: 866ba392f1c1e84b92ad75fa66550f26e8af8e48.","https://github.com/GalaxyGeneralRobotics/Click-and-Traverse"),
        ("[5] Implementation fork: Skvayzer/GRAIL, research/grail-cat-terrain.","https://github.com/Skvayzer/GRAIL/tree/research/grail-cat-terrain"),
        ("[6] Final training run: W&B rmx6awvl (no policy evaluation).","https://wandb.ai/skvayzer/grail-cat/runs/rmx6awvl"),
    ]
    y=1.95
    for label,url in refs:
        y=p.para(y,label,size=9.2)
        y=p.text(.60,y-.05,7.05,url,8.2,TEAL,url=url)+.16
    y=p.heading(y+.04,"Local proposal and data provenance")
    y=p.para(y,"Proposal: ~/research/research/GRAIL-CAT - Terrain-Aware Whole-Body Control Proposal.md (11 September). GRAIL dataset pin: 40e795761302e611c1e7e3a6caefdd010d56c199. Released terrain checkpoint SHA-256: 699b17677fa7e6f98ed4cf3dfb99d2fedb56276116a56f2be593aa756fc00120.",size=8.8)
    y=p.para(y,"Final checkpoint SHA-256: "+d["runs"][-1]["latest"]["sha256"]+". Bank manifest, all six metric/config/status/latest files, existing evaluation summaries, baseline records, source modules and included images are hashed in manifest.json beside this PDF.",size=8.8)
    y=p.heading(y,"Plotting and interpretation")
    y=p.para(y,"No newly generated policy trajectories or evaluations. All photos/rendered scenes are existing artifacts, explicitly labelled as frozen-policy, geometry-only or kinematic. Learning curves use the complete parent-linked production history, without capacity/pilot data or duplicate transitions. Figure 11 uses trailing seven-update means; other learning plots show raw points. Final-window percentages weight by completed episodes, not equally by update.",size=9.2)
    y=p.para(y,"One training lineage with variable batch sizes and resource sharing is not a multi-seed experiment. Failure indicators may overlap. Reported per-step reward is not episode return, action-retention MSE is not terrain retention, and the 45 inherited validation-scene visits do not imply evaluation after --no-eval. Physical uncluttered baselines and native SDF-clutter training have different scope.",size=9.2)
    p.para(y,"Accompanying files: report_data.json (plotted records/summaries), manifest.json (SHA-256 provenance), report_text.md (selectable report text), and page previews. The CPU-only builder is research/build_implementation_report.py. External primary-source pages were checked on 14 September 2026.",size=9.2)
    p.source("All numbers are existing recorded results or explicitly labelled derivations. No claim of final policy evaluation or deployment readiness.")
    p.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists() and not args.overwrite:
        parser.error("Output exists; choose a new directory or --overwrite for this report's files")
    out.mkdir(parents=True,exist_ok=True)
    evidence=Evidence();data=load_data(evidence)
    report=Report(out,evidence,data)
    try: make_report(report)
    finally: report.pdf.close()
    clean_runs=[{k:v for k,v in r.items() if k not in ("config","path","rows")}
                for r in data["runs"]]
    result=dict(final_steps=data["final_steps"],final_updates=data["final_updates"],
        tail_window=data["tail"],runs=clean_runs,metrics=data["rows"],
        existing_evaluations=data["evaluations"],baseline_mpjpe_mm=data["baseline"],
        bank_family_counts={f"{a}/{b}":n for (a,b),n in collections.Counter(
            (s["family"],s["split"]) for s in data["bank"]["scenes"]).items()})
    (out/"report_data.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (out/"report_text.md").write_text("\n\n".join(
        "# "+p["title"]+"\n\n"+"\n\n".join(p["text"][1:]) for p in report.pages)+"\n")
    evidence.register(Path(__file__))
    manifest=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_revision=data["revision"],pages=len(report.pages),policy_evaluation_run=False,
        simulator_started=False,robot_actuation=False,inputs=evidence.files,
        pdf_sha256=hashlib.sha256((out/"GRAIL_CAT_Implementation_Report_20260914.pdf").read_bytes()).hexdigest())
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(pdf=str(out/"GRAIL_CAT_Implementation_Report_20260914.pdf"),
        pages=len(report.pages),bytes=(out/"GRAIL_CAT_Implementation_Report_20260914.pdf").stat().st_size,
        final_steps=data["final_steps"],report_only=True),indent=2))


if __name__=="__main__": main()
