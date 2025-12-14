# POIScale.py (지금까지 된 부분 !!!!1209)
# - INPUT_FOLDER 안의 *.dxf 모두 처리
# - 결과는 같은 폴더에 동일 파일명 + .txt 로 저장 (A.dxf -> A.txt)
# - 라벨은 '한글 1자 이상 포함' 항목만 통과
# - 레이어/마스크/사용자제외/병합/좌표변환 로직까지 구현했음_해원

# pip install ezdxf
import ezdxf, re, math
from pathlib import Path
from ezdxf.math import Vec3

# =========================
# 설정값
# =========================
INPUT_FOLDER = r"C:\Users\yujinjo\cad"   # 처리할 폴더
GLOB_PATTERN = "B1F _2.dxf"                   # 처리할 파일 패턴
PROCESS_RECURSIVELY = False              # 하위 폴더까지 처리하려면 True

# 좌표 보정 옵션
FLIP_Y = False
GLOBAL_ROT_DEG = 0.0
GLOBAL_SCALE = 1.0
OFFSET_X = 0.0
OFFSET_Z = 0.0
ADD_YAW_TO_LABEL = 0.0

# 텍스트 높이 필터 (미터). 0이면 비활성화
MIN_TEXT_HEIGHT_M = 0.0

# 가까운 텍스트 병합 옵션
MERGE_NEARBY = True
MERGE_RADIUS_M = 0.001
MERGE_ROT_TOL_DEG = 25

# =========================
# 제외/허용 텍스트
# =========================
CORRIDOR_KEYWORDS = [
    "복도","코리더","corridor","hallway","홀","대기홀","통로","보행자통로","로비","로비홀",
    "상부 PIT","노동조합","방화구획", "내려옴", "올라감", "평면도", "내려감", "올라옴", "잔액"
]

EXCLUDE_PATTERNS = [
    r"^\s*FL[±\+\-]?\s*[\d\.\,]*\s*$",
    r"^\s*SL[±\+\-]?\s*[\d\.\,]*\s*$",
    r"^\s*SL-\s*\d+\s*$",
    r"^\s*Ø?\s*\d{1,3}(?:[ ,]?\d{3})*(?:\.\d+)?\s*(?:mm|m|M)?\s*$",   # 지름/치수 숫자
    r"^\s*[↔↕→←↑↓⇔⇕]\s*$",                                          # 방향 화살표만
    r".*\*",                                                         # 별표 포함 라인
    r".*/NO(\s|$)",
    r"^\s*[-+]?\d+(?:[.,]\d+)?\s*$",                                 # 숫자만
    r"^[A-Za-z0-9]+(?:[*/\-][A-Za-z0-9]+)+$",                        # 24P-120, D.A(OA) 등
    r"^[\(\[]?\s*[A-Za-z]{1,2}\s*-?\s*\d{1,3}\s*[\)\]]?\s*$",
]
EXCLUDE_RE = re.compile("|".join(EXCLUDE_PATTERNS), re.IGNORECASE)
ALLOW_FULL_RE = re.compile(r"^[가-힣A-Za-z0-9#\-\._/()\s·]+$")

# === 필수 조건: 라벨에 '한글' 1자 이상 포함되어야 함 ===
REQUIRE_HANGUL = True
HANGUL_RE = re.compile(r"[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]")

# =========================
# 레이어 필터 (복도도빼자)
# =========================
LAYER_BLACKLIST = {"DIM","DIMS","DIMENSION","AXIS","GRID","CENTER","CENTERLINE",
                   "ANNOT_GRID","ELEV","LEVEL","SECTION","DETAIL"}
LAYER_WHITELIST = {"POI","ROOM","ROOM_NAME","TEXT","ANNOTATION"}
USE_LAYER_WHITELIST = False

# =========================
# 단위/좌표 변환
# =========================
INSUNITS_TO_M = {0:1.0,1:0.0254,2:0.3048,4:0.001,5:0.01,6:0.1,7:1.0,8:1000.0}
def unit_scale(doc): return INSUNITS_TO_M.get(int(doc.header.get("$INSUNITS", 0)), 1.0)

def to_unity_xz(x_m: float, y_m: float):
    if FLIP_Y: y_m = -y_m
    a = math.radians(GLOBAL_ROT_DEG)
    xr = x_m*math.cos(a) - y_m*math.sin(a)
    zr = x_m*math.sin(a) + y_m*math.cos(a)
    xr = xr*GLOBAL_SCALE + OFFSET_X
    zr = zr*GLOBAL_SCALE + OFFSET_Z
    return xr, zr

# =========================
# 유틸
# =========================
def layer_ok(layer_name: str) -> bool:
    if layer_name in LAYER_BLACKLIST: return False
    return (not USE_LAYER_WHITELIST) or (layer_name in LAYER_WHITELIST)

def text_height_m(ent, base_scale) -> float:
    for key in ("height","char_height"):
        if hasattr(ent.dxf, key):
            val = getattr(ent.dxf, key)
            if val not in (None, 0, 0.0):
                try:
                    return float(val) * base_scale
                except:
                    pass
    return float("inf")

def big_enough(ent, base_scale)->bool:
    if MIN_TEXT_HEIGHT_M <= 0:
        return True
    h = text_height_m(ent, base_scale)
    return (h == float("inf")) or (h >= MIN_TEXT_HEIGHT_M)

def clean_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\\P","\n")
    s = (s.replace("%%d","°").replace("%%D","°")
           .replace("%%p","±").replace("%%P","±")
           .replace("%%c","Ø").replace("%%C","Ø"))
    s = re.sub(r"\\[A-Za-z]+","",s).strip()
    return s

def include_text(txt: str) -> bool:
    if not txt: return False
    if REQUIRE_HANGUL and not HANGUL_RE.search(txt):  # 1) 한글 필수
        return False
    if EXCLUDE_RE.match(txt):                         # 2) 제외 패턴
        return False
    if not ALLOW_FULL_RE.fullmatch(txt):              # 3) 위생 필터
        return False
    t = txt.replace(" ","").lower()                   # 4) 통로성 키워드 제외
    if any(k.replace(" ","").lower() in t for k in CORRIDOR_KEYWORDS):
        return False
    return True

# =========================
# 외곽/방 마스크 (필터링해야지댐)
# =========================
OUTLINE_LAYERS = {"WALL","WALLS","A-WALL","A-WALL-OUT","OUTLINE","외곽","벽체"}
ROOM_POLY_LAYERS = set()

def polygon_area(poly):
    a=0.0
    for i in range(len(poly)):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%len(poly)]
        a += x1*y2 - x2*y1
    return abs(a)/2.0

def point_in_polygon(x, y, poly):
    inside=False
    n=len(poly)
    for i in range(n):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%n]
        if ((y1>y)!=(y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12)+x1):
            inside = not inside
    return inside

def collect_closed_polylines(msp, wanted_layers, base_scale):
    polys=[]
    if not wanted_layers: return polys
    for e in msp.query("LWPOLYLINE"):
        if not e.closed: continue
        if e.dxf.layer not in wanted_layers: continue
        pts=[(p[0]*base_scale, p[1]*base_scale) for p in e.get_points("xy")]
        if len(pts)>=3: polys.append(pts)
    for e in msp.query("POLYLINE"):
        if not getattr(e, "is_closed", False): continue
        if e.dxf.layer not in wanted_layers: continue
        pts=[(v.dxf.location.x*base_scale, v.dxf.location.y*base_scale) for v in e.vertices]
        if len(pts)>=3: polys.append(pts)
    return polys

def load_masks(msp, base_scale):
    outer = collect_closed_polylines(msp, OUTLINE_LAYERS, base_scale)
    outer = sorted(outer, key=polygon_area, reverse=True)[:1]
    rooms = collect_closed_polylines(msp, ROOM_POLY_LAYERS, base_scale)
    return outer, rooms

def inside_building_or_room(x_m, y_m, outer_polys, room_polys):
    if room_polys:
        return any(point_in_polygon(x_m, y_m, rp) for rp in room_polys)
    if outer_polys:
        return any(point_in_polygon(x_m, y_m, op) for op in outer_polys)
    return True

# =========================
# 사용자 제외 폴리곤(사각형/자유형) (건드리지말기)
# =========================
EXCLUDE_POLYS_RAW = [
    [
        (80829.9738, 108035.9364),
        (85090.5927,  53983.7750),
        (144674.2029,  51548.2361),
        (144615.6532, 106084.1921),
    ],
    [
        (110287.3270,   69788.4459),
        (106917.0557,   28846.2314),
        (317343.6533,   14854.0230),
        (313795.9989,   67494.0141),
    ],
    [
        (229664.2732,   97378.7573),
        (230100.8319,   52818.6864),
        (311781.2572,   52235.9197),
        (285766.8447,   96962.9697),
    ],
    [
        (235356.9103, 116112.9814),
        (235356.9103,  96675.7487),
        (275864.8750,   97681.9118),
        (278473.8623, 115930.0438),
    ],
    [
        (138175.4652, 211958.2817),
        (140638.4575, 163558.6078),
        (199887.1443, 162601.5530),
        (195371.6551, 212231.7264),
    ],
]

def normalize_polys(raw, base_scale):
    polys=[]
    for poly in raw:
        polys.append([(x*base_scale, y*base_scale) for (x,y) in poly])
    return polys

# =========================
# 각도/거리 기반 제외영역(평행사변형)
# =========================
def pol2xy(r, deg):
    a = math.radians(deg)
    return r*math.cos(a), r*math.sin(a)

EXCLUDE_PARALLELOGRAMS_POLAR = [
    {
        "anchor": (44664.5994, 167529.734),
        "v1": (95099.1566, 315.0),
        "v2": (27924.0676, 222.0),
        "diag": (95981.3365, 298.0),
    },
    {
        "anchor": (75589.5558, 178937.1613),
        "v1": (61915.0028, 226.0),
        "v2": (32555.0749, 134.0),
        "diag": (71211.7827, 188.0),
    },
]

def build_polys_from_polar(specs, base_scale):
    polys=[]
    for s in specs:
        ax, ay = s["anchor"]
        d1, a1 = s["v1"];   dx1, dy1 = pol2xy(d1, a1)
        d2, a2 = s["v2"];   dx2, dy2 = pol2xy(d2, a2)

        p0 = (ax*base_scale,               ay*base_scale)
        p1 = ((ax+dx1)*base_scale,         (ay+dy1)*base_scale)
        p2 = ((ax+dx1+dx2)*base_scale,     (ay+dy1+dy2)*base_scale)
        p3 = ((ax+dx2)*base_scale,         (ay+dy2)*base_scale)
        polys.append([p0, p1, p2, p3])

        if "diag" in s:
            dd, ad = s["diag"]
            ddx, ddy = pol2xy(dd, ad)
            ex, ey = ax+ddx, ay+ddy
            if abs((ax+dx1+dx2)-ex) > 5 or abs((ay+dy1+dy2)-ey) > 5:
                print("[WARN] polar diag mismatch (>5 units):", s)
    return polys

def in_any_exclude_poly(x_m, y_m, polys):
    return any(point_in_polygon(x_m, y_m, poly) for poly in polys)

# =========================
# 병합 ㄱㄱ
# =========================
def _ang_delta(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)

def _uniq_join_keep_order(names):
    seen = set(); out = []
    for n in names:
        if n and n not in seen:
            out.append(n); seen.add(n)
    return " ".join(out)

# =========================
# 한 파일 처리
# =========================
def process_one(dxf_path: Path) -> bool:
    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception as e:
        print(f"[ERROR] readfile failed: {dxf_path.name}: {e}")
        return False

    msp = doc.modelspace()
    base_scale = unit_scale(doc)

    base_scale = 1.0
    outer_polys, room_polys = load_masks(msp, base_scale)
    EXCL_POLYS  = normalize_polys(EXCLUDE_POLYS_RAW, base_scale)
    EXCL_POLYS += build_polys_from_polar(EXCLUDE_PARALLELOGRAMS_POLAR, base_scale)

    poi_raw = []  # 원시 수집(미터 단위로 맞춰야댐)

    def add_point_raw(name, x, y, z, rot, h):
        x_m, y_m, z_m = x*base_scale, y*base_scale, z*base_scale
        if in_any_exclude_poly(x_m, y_m, EXCL_POLYS):   # 사용자 제외
            return
        if not inside_building_or_room(x_m, y_m, outer_polys, room_polys):  # 마스크
            return
        poi_raw.append({
            "name": name,
            "x": x_m, "y": y_m, "z": z_m,
            "rot": (rot + ADD_YAW_TO_LABEL) % 360.0,
            "h": (h if h != float("inf") else 1.0),  # 가중치
        })

    # --- TEXT
    for e in msp.query("TEXT"):
        if not layer_ok(e.dxf.layer): continue
        if not big_enough(e, base_scale): continue
        name = clean_text(e.dxf.text)
        if not include_text(name): continue
        h = text_height_m(e, base_scale)
        add_point_raw(name, *e.dxf.insert, float(e.dxf.rotation or 0.0), h)

    # --- MTEXT
    for e in msp.query("MTEXT"):
        if not layer_ok(e.dxf.layer): continue
        if not big_enough(e, base_scale): continue
        name = clean_text(e.text)
        if not include_text(name): continue
        x,y,z = e.dxf.insert
        try: rot = float(e.get_rotation())
        except: rot = 0.0
        h = text_height_m(e, base_scale)
        add_point_raw(name, x,y,z, rot, h)

    # --- INSERT의 ATTRIB
    for ins in msp.query("INSERT"):
        if not layer_ok(ins.dxf.layer): continue
        bx,by,bz = ins.dxf.insert
        rot = float(ins.dxf.rotation or 0.0)
        a = math.radians(rot)
        sx = float(getattr(ins.dxf,"xscale",1.0) or 1.0)
        sy = float(getattr(ins.dxf,"yscale",1.0) or 1.0)
        for att in getattr(ins, "attribs", []):
            if not layer_ok(att.dxf.layer): continue
            if not big_enough(att, base_scale): continue
            name = clean_text(att.dxf.text)
            if not include_text(name): continue
            ax,ay,az = att.dxf.insert
            rx = (ax*math.cos(a) - ay*math.sin(a))*sx
            ry = (ax*math.sin(a) + ay*math.cos(a))*sy
            h = text_height_m(att, base_scale) * max(abs(sx), abs(sy))
            add_point_raw(name, bx+rx, by+ry, bz+az, rot, h)

    # --- 블록 내부 TEXT/MTEXT (월드좌표로 변환해야대나?-해원)
    for br in msp.query("INSERT"):
        try:
            M = br.matrix44()
            br_rot = float(br.dxf.rotation or 0.0)
            block = br.block()
            sx = float(getattr(br.dxf,"xscale",1.0) or 1.0)
            sy = float(getattr(br.dxf,"yscale",1.0) or 1.0)
            scl = max(abs(sx), abs(sy))
        except Exception:
            continue

        for e in block.entity_space:
            dxt = e.dxftype()
            if dxt not in ("TEXT","MTEXT"): continue

            lyr = getattr(e.dxf, "layer", "")
            if lyr == "0": lyr = br.dxf.layer  # 레이어 0 상속
            if not layer_ok(lyr): continue
            if not big_enough(e, base_scale): continue

            if dxt == "TEXT":
                name = clean_text(e.dxf.text)
                rot_local = float(getattr(e.dxf, "rotation", 0.0) or 0.0)
                x,y,z = e.dxf.insert
            else:
                name = clean_text(e.text)
                try: rot_local = float(e.get_rotation())
                except: rot_local = 0.0
                x,y,z = e.dxf.insert

            if not include_text(name): continue

            p_world = M.transform(Vec3(x,y,z))
            rot_world = br_rot + rot_local
            h = text_height_m(e, base_scale) * scl
            add_point_raw(name, p_world.x, p_world.y, p_world.z, rot_world, h)

    # -------------------------
    # 병합
    # -------------------------
    def merge_poi_groups(raw):
        if not MERGE_NEARBY:
            out = []
            for r in raw:
                X, Z = to_unity_xz(r["x"], r["y"])
                out.append((r["name"], X, Z, r["z"], r["rot"]))
            return out

        n = len(raw)
        used = [False]*n
        merged = []

        for i in range(n):
            if used[i]:
                continue
            used[i] = True
            grp = [i]

            xi, yi, ri = raw[i]["x"], raw[i]["y"], raw[i]["rot"]
            for j in range(i+1, n):
                if used[j]:
                    continue
                dx = raw[j]["x"] - xi
                dy = raw[j]["y"] - yi
                if math.hypot(dx, dy) <= MERGE_RADIUS_M and _ang_delta(ri, raw[j]["rot"]) <= MERGE_ROT_TOL_DEG:
                    grp.append(j)
                    used[j] = True

            # 대표 회전: 가장 큰 글자 높이
            k_rot = max(grp, key=lambda k: raw[k]["h"] if raw[k]["h"] else 1.0)
            rot_sel = raw[k_rot]["rot"]

            # 대표 위치: 가중 평균
            wsum = 0.0; xs=ys=zs=0.0
            for k in grp:
                w = raw[k]["h"] if raw[k]["h"] else 1.0
                xs += raw[k]["x"]*w
                ys += raw[k]["y"]*w
                zs += raw[k]["z"]*w
                wsum += w
            cx = xs/wsum if wsum>0 else raw[i]["x"]
            cy = ys/wsum if wsum>0 else raw[i]["y"]
            cz = zs/wsum if wsum>0 else raw[i]["z"]

            # 읽기방향 정렬
            a = math.radians(rot_sel)
            c, s = math.cos(a), math.sin(a)
            def to_local(k):
                dx = raw[k]["x"] - cx
                dy = raw[k]["y"] - cy
                x_p =  dx*c + dy*s
                y_p = -dx*s + dy*c
                return x_p, y_p
            order = sorted(grp, key=lambda k: (-to_local(k)[1], to_local(k)[0]))

            # 중복 제거+순서 유지 결합
            names_sorted = [raw[k]["name"] for k in order]
            name_joined = _uniq_join_keep_order(names_sorted)

            X, Z = to_unity_xz(cx, cy)
            merged.append((name_joined, X, Z, cz, rot_sel))

        return merged

    poi = merge_poi_groups(poi_raw)

    # 저장: 같은 폴더, 파일명만 .txt 로 교체
    out_path = dxf_path.with_suffix(".txt")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for name,X,Z,Y,rot in poi:
                f.write(f"{name}|{X:.6f}|{Z:.6f}|{Y:.6f}|{rot:.3f}\n")
        print(f"[OK] {dxf_path.name}: raw {len(poi_raw)} -> merged {len(poi)} -> {out_path.name}")
        return True
    except Exception as e:
        print(f"[ERROR] write failed: {out_path}: {e}")
        return False

# =========================
# 메인 (수정완료) 1207
# =========================
def main():
    root = Path(INPUT_FOLDER).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] 폴더가 아닙니다: {root}")
        return

    if PROCESS_RECURSIVELY:
        dxf_files = sorted(root.rglob(GLOB_PATTERN))
    else:
        dxf_files = sorted(root.glob(GLOB_PATTERN))

    if not dxf_files:
        print(f"[WARN] DXF 없음: {root} / pattern={GLOB_PATTERN}")
        return

    ok = 0
    for dxf in dxf_files:
        ok += 1 if process_one(dxf) else 0

    print(f"[DONE] {ok}/{len(dxf_files)} 파일 완료 (출력은 같은 폴더 .txt)")

if __name__ == "__main__":
    main()
