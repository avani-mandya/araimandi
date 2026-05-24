import streamlit as st
import numpy as np
import time, os, urllib.request

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

IS_CLOUD = os.path.exists("/mount/src") or not CV2_OK

IS_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") is not None or os.path.exists("/mount/src")

st.set_page_config(page_title="Araimandi Guru", layout="wide", page_icon="🪷")

# ── Session state ─────────────────────────────────────────────
for key, val in {
    "stage":       "preview",   # preview | capturing | result
    "result_img":  None,
    "scores":      None,
    "active_tab":  "home",      # home | analyser | howto
    "cancelled":   False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

GREEN  = (0, 220, 80);   YELLOW = (255, 255, 0)
WHITE  = (255, 255, 255); RED    = (255, 80, 80)
ORANGE = (255, 165, 0);   BLACK  = (0, 0, 0)

# ── Pose reference image ──────────────────────────────────────
def make_pose_ref():
    """Load the actual Araimandi reference diagram"""
    try:
        img = Image.open("araimandi_reference.png")
        return img
    except:
        pass
    # Fallback stick figure if image not found
    return make_pose_ref_fallback()

def make_pose_ref_fallback():
    W, H = 300, 420
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (22, 20, 18)
    G = (80, 200, 80); Y = (0, 220, 220); WH = (220, 215, 200); GO = (76, 168, 201)
    cv2.circle(img, (150,45), 28, WH, 2)
    cv2.line(img, (150,73),(150,100), WH, 2)
    cv2.line(img, (95,110),(205,110), WH, 2)
    cv2.line(img, (95,110),(70,160),  WH, 2)
    cv2.line(img, (205,110),(230,160),WH, 2)
    cv2.line(img, (70,160),(55,210),  WH, 2)
    cv2.line(img, (230,160),(245,210),WH, 2)
    cv2.line(img, (150,100),(150,210),WH, 3)
    cv2.line(img, (110,210),(190,210),GO, 3)
    cv2.line(img, (110,210),(55,300), G,  3)
    cv2.line(img, (55,300),(80,370),  G,  3)
    cv2.line(img, (190,210),(245,300),G,  3)
    cv2.line(img, (245,300),(220,370),G,  3)
    cv2.circle(img,(150,210),7,GO,-1)
    cv2.circle(img,(55,300), 9,G,-1)
    cv2.circle(img,(245,300),9,G,-1)
    cv2.circle(img,(80,370), 7,Y,-1)
    cv2.circle(img,(220,370),7,Y,-1)
    for a,b in [(150,210),(55,300),(150,370),(245,300)]:
        pass
    pts = [(150,210),(55,300),(150,370),(245,300)]
    for i in range(4):
        cv2.line(img, pts[i], pts[(i+1)%4], G, 1)
    cv2.line(img,(20,372),(280,372),Y,3)
    cv2.putText(img,"~90-110deg",(2,295),cv2.FONT_HERSHEY_SIMPLEX,0.32,G,1)
    cv2.putText(img,"~90-110deg",(205,295),cv2.FONT_HERSHEY_SIMPLEX,0.32,G,1)
    cv2.putText(img,"Spine straight",(95,155),cv2.FONT_HERSHEY_SIMPLEX,0.32,WH,1)
    cv2.putText(img,"Hips",(155,225),cv2.FONT_HERSHEY_SIMPLEX,0.38,GO,1)
    cv2.putText(img,"180deg baseline",(60,395),cv2.FONT_HERSHEY_SIMPLEX,0.35,Y,1)
    return img

# ── MediaPipe ─────────────────────────────────────────────────
@st.cache_resource(ttl=3600)
def load_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    model_path = "pose_landmarker.task"
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            model_path)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1, min_pose_detection_confidence=0.4)
    return mp_vision.PoseLandmarker.create_from_options(opts)

def detect(rgb, detector):
    try:
        import mediapipe as mp
        H,W = rgb.shape[:2]
        r = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not r.pose_landmarks: return None
        lm = r.pose_landmarks[0]
        def px(i): return (int(lm[i].x*W), int(lm[i].y*H))
        return {k:px(i) for k,i in zip(
            ['l_hip','r_hip','l_knee','r_knee','l_ankle','r_ankle','l_heel','r_heel','l_foot','r_foot'],
            [23,24,25,26,27,28,29,30,31,32])}
    except: return None

def knee_angle(h,k,a):
    v1=np.array([h[0]-k[0],h[1]-k[1]],dtype=float)
    v2=np.array([a[0]-k[0],a[1]-k[1]],dtype=float)
    n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
    if n1==0 or n2==0: return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))))

def rate(a):
    if  80<=a<=110: return 100,"Perfect ✓"
    if  70<=a<=125: return 80,"Good"
    if  60<=a<=140: return 55,"Needs work"
    if  50<=a<=155: return 30,"Too straight"
    return 10,"Not in Araimandi"

def col(s): return GREEN if s>=80 else (ORANGE if s>=55 else RED)

def draw_frame(img, j, scored=False):
    H,W=img.shape[:2]
    lh=j['l_hip']; rh=j['r_hip']
    lk=j['l_knee']; rk=j['r_knee']
    la=j['l_ankle']; ra=j['r_ankle']
    lf=j['l_foot']; rf=j['r_foot']
    hm=((lh[0]+rh[0])//2,(lh[1]+rh[1])//2)
    fm=((lf[0]+rf[0])//2,(lf[1]+rf[1])//2)
    fy=min((lf[1]+rf[1])//2,H-8)
    la_=knee_angle(lh,lk,la); ra_=knee_angle(rh,rk,ra)
    ls,_=rate(la_); rs,_=rate(ra_)
    fd=abs(lf[1]-rf[1]); fs=100 if fd<=15 else max(0,int(100-fd*1.5))
    lc=col(ls) if scored else GREEN; rc2=col(rs) if scored else GREEN
    cv2.line(img,hm,lk,lc,4,cv2.LINE_AA); cv2.line(img,hm,rk,rc2,4,cv2.LINE_AA)
    cv2.line(img,lk,fm,lc,4,cv2.LINE_AA); cv2.line(img,rk,fm,rc2,4,cv2.LINE_AA)
    for pt,c in [(hm,GREEN),(lk,lc),(rk,rc2),(fm,YELLOW),(lf,YELLOW),(rf,YELLOW)]:
        cv2.circle(img,pt,9,c,-1,cv2.LINE_AA); cv2.circle(img,pt,9,WHITE,1,cv2.LINE_AA)
    cv2.line(img,(0,fy),(W,fy),YELLOW,6,cv2.LINE_AA)
    cv2.circle(img,(lf[0],fy),10,YELLOW,-1,cv2.LINE_AA); cv2.circle(img,(lf[0],fy),10,WHITE,2,cv2.LINE_AA)
    cv2.circle(img,(rf[0],fy),10,YELLOW,-1,cv2.LINE_AA); cv2.circle(img,(rf[0],fy),10,WHITE,2,cv2.LINE_AA)
    lbl="180 deg FEET BASELINE"
    (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
    lx=W//2-tw//2; ly=max(fy-8,22)
    cv2.rectangle(img,(lx-5,ly-th-4),(lx+tw+5,ly+4),BLACK,-1)
    cv2.putText(img,lbl,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,0.55,YELLOW,2,cv2.LINE_AA)
    if scored:
        for pt,ang,sc,side in [(lk,la_,ls,"L"),(rk,ra_,rs,"R")]:
            c2=col(sc); dx=-115 if side=="L" else 14
            cv2.rectangle(img,(pt[0]+dx-3,pt[1]-30),(pt[0]+dx+110,pt[1]-8),BLACK,-1)
            cv2.putText(img,f"{side}: {ang:.0f}deg",(pt[0]+dx,pt[1]-12),cv2.FONT_HERSHEY_SIMPLEX,0.55,c2,2,cv2.LINE_AA)
        ov=(ls+rs+fs)//3; oc=col(ov)
        cv2.rectangle(img,(8,8),(230,55),BLACK,-1)
        cv2.putText(img,f"Score: {ov}%",(14,44),cv2.FONT_HERSHEY_SIMPLEX,1.1,oc,2,cv2.LINE_AA)
    else:
        cv2.rectangle(img,(0,H-42),(W,H),BLACK,-1)
        cv2.putText(img,f"L knee: {la_:.0f}deg  |  R knee: {ra_:.0f}deg  |  Target: 80-110deg",
                    (10,H-12),cv2.FONT_HERSHEY_SIMPLEX,0.55,WHITE,1,cv2.LINE_AA)
    return img, la_, ra_, ls, rs, fs

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:1rem 0 0.5rem'>
<h1 style='font-family:Georgia,serif;color:#c9a84c;font-size:2.8rem;margin:0'>
🪷 Araimandi Guru
</h1>
<p style='color:#888;font-size:1rem;margin-top:4px'>
AI-powered Bharatanatyam posture analyser
</p></div>
""", unsafe_allow_html=True)

tab_home, tab_analyser = st.tabs(["🏠 Home", "📷 Analyser"])

# ══════════════════════════════════════════════════════════════
# TAB 1: HOME
# ══════════════════════════════════════════════════════════════
with tab_home:
    st.markdown("---")
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("""
## What is Araimandi?
**Araimandi** is the foundational half-sitting posture in **Bharatanatyam**, India's classical dance form.

It is the base position from which most movements originate. Mastering Araimandi requires:

- 🦵 **Knees bent and pushed wide outward** — forming a diamond shape
- 🦶 **Heels flat on the floor** — feet at 180° to each other
- 🧍 **Spine perfectly upright** — chest open, shoulders relaxed
- 🔄 **Turnout from the hips** — not from the knees

### Why does it matter?
Incorrect Araimandi leads to **knee and lower back injuries** over time. This app helps dancers check their form using AI.

### How this app works
The AI detects your body joints automatically and draws a **green diamond** on your lower body. It measures the **knee angle** and checks if your **feet are level** at 180°.
""")

        st.markdown("---")
        st.markdown("### 📖 How to use")
        st.markdown("""
**Step 1** — Place camera at hip height, stand 2–3 metres back so full body is visible

**Step 2** — Click **📷 Analyser** tab — green diamond appears on your joints

**Step 3** — Get into Araimandi, press **▶ Start 45s Timer**

**Step 4** — Hold still — app auto-captures and scores!

**Step 5** — Press **❌ Cancel & Retake** anytime to restart
""")
        st.markdown("---")
        st.markdown("""
| ❌ Mistake | ✅ Correction |
|-----------|--------------|
| Knees point forward | Push knees outward over toes |
| Heels lifted | Keep both heels flat on floor |
| Leaning forward | Lift chest, engage core |
| One side higher | Even out both sides |
""")

        st.info("👆 Click the **📷 Analyser** tab above to start!")

    with c2:
        st.markdown("### Correct Araimandi Pose")
        st.image(make_pose_ref(), caption="Reference: correct Araimandi alignment", use_container_width=True)
        st.markdown("""
<div style='background:#1a1814;border:1px solid #333;border-radius:8px;padding:1rem'>
<b style='color:#c9a84c'>Key checkpoints:</b><br><br>
🟢 <b>Hips</b> — top of diamond<br>
🟢 <b>Knees</b> — sides, pushed wide outward<br>
🟢 <b>Feet</b> — bottom of diamond<br>
🟡 <b>Yellow line</b> — 180° floor baseline<br>
📐 <b>Knee angle target</b> — 90–110°
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2: ANALYSER
# ══════════════════════════════════════════════════════════════
with tab_analyser:
    st.markdown("---")
    if IS_CLOUD:
        st.warning("⚠️ Camera not available on web version. Run the app locally to use the AI analyser.")
        st.markdown("### Araimandi Reference Guide")
        c1, c2 = st.columns([1,1])
        with c1:
            st.image(make_pose_ref(), caption="Correct Araimandi pose", use_container_width=True)
        with c2:
            st.markdown("""
**Key checkpoints:**
- 🟢 Knees pushed wide outward
- 🟡 Heels flat on 180° baseline
- 🧍 Spine perfectly upright
- 📐 Knee angle: 80–110°

**Scoring:**
| Knee angle | Score |
|-----------|-------|
| 80–110° | 💯 Perfect |
| 70–125° | 👍 Good |
| 60–140° | ⚠️ Needs work |
| >140° | ❌ Too straight |
""")
        st.stop()

    # ── RESULT VIEW ──
    if st.session_state["stage"] == "result":
        s  = st.session_state["scores"]
        ov = s["overall"]
        if ov>=80:   st.success(f"### 🏆 {ov}% — Excellent Araimandi!")
        elif ov>=60: st.warning(f"### 👍 {ov}% — Good! Keep practising.")
        else:        st.error(f"### 💪 {ov}% — Needs more work.")

        st.markdown("---")
        ic, sc = st.columns([2,1])
        with ic:
            st.image(st.session_state["result_img"],
                     caption="Your captured Araimandi pose", use_container_width=True)
        with sc:
            st.markdown("### 📊 Breakdown")
            for label, sc2 in [
                (f"Left knee ({s['l_ang']:.0f}°)",  s['l_sc']),
                (f"Right knee ({s['r_ang']:.0f}°)", s['r_sc']),
                ("Feet level (180° baseline)",       s['ft_sc']),
            ]:
                icon = "✅" if sc2>=80 else "⚠️" if sc2>=55 else "❌"
                st.markdown(f"{icon} **{label}**: {sc2}%")
                st.progress(sc2/100)

            st.markdown("---")
            st.markdown("### 🔧 How to improve")
            for f in s["feedback"]:
                st.markdown(f"- {f}")

            st.markdown("---")
            st.markdown("### Reference pose")
            st.image(make_pose_ref(), use_container_width=True)

            st.markdown("---")
            if st.button("🔄 Try Again", type="primary", use_container_width=True):
                st.session_state["stage"]  = "preview"
                st.session_state["scores"] = None
                st.rerun()

    # ── CAPTURING VIEW (45s timer running) ──
    elif st.session_state["stage"] == "capturing":
        st.markdown("""
<div style='background:#1a1814;border-left:4px solid #e67e22;padding:1rem;border-radius:4px;margin-bottom:1rem'>
⏱ <b style='color:#e67e22'>Timer running — get into Araimandi and hold still!</b>
The app will auto-capture when the timer ends.
</div>
""", unsafe_allow_html=True)

        cancel_btn = st.button("❌ Cancel & Retake", use_container_width=True)
        frame_win  = st.empty()
        status_win = st.empty()

        if cancel_btn:
            st.session_state["stage"]     = "preview"
            st.session_state["cancelled"] = True
            st.session_state.pop("deadline", None)
            st.session_state.pop("capture_buf", None)
            st.rerun()

        deadline = st.session_state.get("deadline", time.time() + 45)
        detector = load_detector()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        buf = []
        last_rgb = None

        # Run while loop for entire countdown — no blinking
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_rgb = rgb.copy()
            H, W = rgb.shape[:2]
            j = detect(rgb, detector)
            if j:
                buf.append(j)
                img, la, ra, *_ = draw_frame(rgb.copy(), j, scored=False)
            else:
                img = rgb.copy()
            secs_left = int(np.ceil(deadline - time.time()))
            if secs_left <= 5:
                ov2 = img.copy()
                cv2.rectangle(ov2,(0,0),(W,H),(15,13,10),-1)
                cv2.addWeighted(ov2,0.4,img,0.6,0,img)
                cv2.putText(img,str(max(secs_left,1)),(W//2-50,H//2+50),
                            cv2.FONT_HERSHEY_SIMPLEX,7,(76,168,201),10,cv2.LINE_AA)
                cv2.putText(img,"Hold still!",(W//2-90,H//2+115),
                            cv2.FONT_HERSHEY_SIMPLEX,1.0,WHITE,2,cv2.LINE_AA)
            frame_win.image(img, use_container_width=True)
            status_win.info(f"⏱ Capturing in **{secs_left}** seconds — hold your Araimandi!")

        cap.release()
        secs_left = 0

        if True:
            # Time's up — process results
            st.session_state.pop("deadline", None)
            st.session_state.pop("capture_buf", None)

            if len(buf) == 0 or last_rgb is None:
                status_win.error("❌ Pose not detected. Try again.")
                st.session_state["stage"] = "preview"
                st.rerun()
            else:
                recent = buf[-10:] if len(buf)>=10 else buf
                keys   = ['l_hip','r_hip','l_knee','r_knee','l_ankle','r_ankle','l_heel','r_heel','l_foot','r_foot']
                stable = {k:(int(round(np.mean([f[k][0] for f in recent]))),
                            int(round(np.mean([f[k][1] for f in recent])))) for k in keys}
                H,W = last_rgb.shape[:2]
                kspan = abs(stable['r_knee'][0]-stable['l_knee'][0])
                a_in  = stable['l_foot'][1]<H-5 and stable['r_foot'][1]<H-5

                if not a_in:
                    status_win.error("❌ Feet cut off — step back and try again")
                    st.session_state["stage"] = "preview"
                    st.rerun()
                elif kspan < 30:
                    status_win.error("❌ Knees too close — step back and try again")
                    st.session_state["stage"] = "preview"
                    st.rerun()
                else:
                    result_img = last_rgb.copy()
                    result_img, l_ang, r_ang, l_sc, r_sc, ft_sc = draw_frame(result_img, stable, scored=True)
                    overall = (l_sc+r_sc+ft_sc)//3
                    feedback = []
                    if l_ang>110: feedback.append(f"🦵 Left knee {l_ang:.0f}° — push knee further outward (target 80–110°)")
                    elif l_ang<70: feedback.append(f"🦵 Left knee {l_ang:.0f}° — sit a little higher")
                    if r_ang>110: feedback.append(f"🦵 Right knee {r_ang:.0f}° — push knee further outward (target 80–110°)")
                    elif r_ang<70: feedback.append(f"🦵 Right knee {r_ang:.0f}° — sit a little higher")
                    if abs(stable['l_foot'][1]-stable['r_foot'][1])>15:
                        feedback.append("🦶 Feet uneven — keep both heels flat on the floor")
                    if abs(l_ang-r_ang)>10:
                        feedback.append("⚖️ Uneven sides — even out both knees")
                    if not feedback: feedback.append("✨ Perfect Araimandi! Excellent form!")

                    st.session_state["result_img"] = result_img
                    st.session_state["scores"] = {
                        "l_ang":l_ang,"r_ang":r_ang,
                        "l_sc":l_sc,"r_sc":r_sc,
                        "ft_sc":ft_sc,"overall":overall,"feedback":feedback}
                    st.session_state["stage"] = "result"
                    st.rerun()
        else:
            status_win.info(f"⏱ Capturing in **{secs_left}** seconds — hold your Araimandi!")
            time.sleep(0.15)
            st.rerun()

    # ── PREVIEW VIEW ──
    else:
        st.markdown("""
<div style='background:#1a1814;border-left:4px solid #c9a84c;padding:1rem;border-radius:4px;margin-bottom:1rem'>
<b style='color:#c9a84c'>Before you start:</b>
Place the camera at hip height, stand <b>2–3 metres back</b> so your full body from head to feet is visible.
When ready, press <b>Start 45s Timer</b>, walk into Araimandi position and hold still.
</div>
""", unsafe_allow_html=True)

        lc, rc = st.columns([3,1])
        with rc:
            st.markdown("### Reference pose")
            st.image(make_pose_ref(), use_container_width=True)
            st.markdown("""
**Scoring:**
| Knee angle | Score |
|-----------|-------|
| 80–110° | 💯 Perfect |
| 70–125° | 👍 Good |
| 60–140° | ⚠️ Needs work |
| >140° | ❌ Too straight |
""")

        with lc:
            start_btn  = st.button("▶ Start 45s Timer — Get into Araimandi!",
                                   type="primary", use_container_width=True)
            frame_win  = st.empty()
            status_win = st.empty()

            if st.session_state.get("cancelled"):
                status_win.warning("↩ Capture cancelled — press Start Timer when ready.")
                st.session_state["cancelled"] = False

        if start_btn:
            st.session_state["stage"]       = "capturing"
            st.session_state["deadline"]    = time.time() + 45
            st.session_state["capture_buf"] = []
            st.rerun()

        # Live preview — while loop, no page rerun = no blinking
        detector = load_detector()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    status_win.error("❌ Camera not found — close other apps using the camera")
                    break
                frame = cv2.flip(frame,1)
                rgb   = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                H,W   = rgb.shape[:2]
                j = detect(rgb, detector)
                if j:
                    img,la,ra,*_ = draw_frame(rgb.copy(), j, scored=False)
                    kspan = abs(j['r_knee'][0]-j['l_knee'][0])
                    a_ok  = j['l_foot'][1]<H-5 and j['r_foot'][1]<H-5
                    if not a_ok:
                        status_win.warning("⚠️ Step back — feet not fully visible")
                    elif kspan<30:
                        status_win.warning("⚠️ Step back — knees too close")
                    else:
                        status_win.success(f"✅ Pose detected — L: {la:.0f}°  R: {ra:.0f}°  (target 80–110°) — press Start Timer!")
                else:
                    img = rgb.copy()
                    cv2.putText(img,"Step back — full body must be visible",
                                (20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,RED,2,cv2.LINE_AA)
                    status_win.error("❌ No pose — step back so full body (head to feet) is visible")
                frame_win.image(img, use_container_width=True)
        finally:
            cap.release()
