"""Builds Calculus_LM.xlsx from mini_weights.json. Every model value is a live formula."""
import json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.comments import Comment
from openpyxl.utils import get_column_letter as L

w = json.load(open("mini_weights.json"))
V, D, T = 10, 4, 16
SHOWN = ["(space)" if c == " " else c for c in w["vocab"]]

ARIAL = "Arial"
BLUE, BLACK, GREEN = "0000FF", "000000", "008000"
f_title = Font(name=ARIAL, size=16, bold=True, color="1F3864")
f_h = Font(name=ARIAL, size=11, bold=True, color="FFFFFF")
f_b = Font(name=ARIAL, size=10, bold=True)
f_n = Font(name=ARIAL, size=10)
f_in = Font(name=ARIAL, size=10, color=BLUE)
f_link = Font(name=ARIAL, size=10, color=GREEN)
f_note = Font(name=ARIAL, size=9, italic=True, color="595959")
fill_h = PatternFill("solid", fgColor="1F3864")
fill_yellow = PatternFill("solid", fgColor="FFFF00")
GROUP = {"P": "DDEBF7", "I": "E2EFDA", "V": "FFF2CC", "A": "FCE4D6", "F": "EDE1F5", "Z": "D9D9D9",
         "S": "DDEBF7", "Pr": "E2EFDA", "R": "FFF2CC"}
thin = Side(style="thin", color="BFBFBF")
box = Border(left=thin, right=thin, top=thin, bottom=thin)
wrap = Alignment(wrap_text=True, vertical="top")
center = Alignment(horizontal="center", vertical="center", wrap_text=True)
NUM = "0.000"

wb = Workbook()


def cell(ws, ref, value, font=f_n, fill=None, fmt=None, align=None, border=False):
    c = ws[ref]
    c.value = value
    c.font = font
    if fill: c.fill = PatternFill("solid", fgColor=fill) if isinstance(fill, str) else fill
    if fmt: c.number_format = fmt
    if align: c.alignment = align
    if border: c.border = box
    return c


def header(ws, row, col, text, fill=None, width=None):
    c = ws.cell(row=row, column=col, value=text)
    c.font = f_b if fill else f_h
    c.fill = PatternFill("solid", fgColor=fill) if fill else fill_h
    c.alignment = center
    c.border = box
    if width: ws.column_dimensions[L(col)].width = width
    return c


def group(ws, row, c1, c2, text, color):
    ws.merge_cells(start_row=row, start_column=c1, end_row=row, end_column=c2)
    c = ws.cell(row=row, column=c1, value=text)
    c.font = f_b; c.fill = PatternFill("solid", fgColor=color); c.alignment = center
    for cc in range(c1, c2 + 1):
        ws.cell(row=row, column=cc).border = box
        ws.cell(row=row, column=cc).fill = PatternFill("solid", fgColor=color)


# =========================== Start Here ===========================
ws0 = wb.active
ws0.title = "Start Here"

# =========================== Settings ===========================
st = wb.create_sheet("Settings")
st.column_dimensions["A"].width = 30
for c in range(2, 19): st.column_dimensions[L(c)].width = 9.5
cell(st, "A1", "Settings & Learned Weights", f_title)
cell(st, "A2", "Blue numbers are inputs. The weights were learned by training in Python (excel/train_mini.py) "
     "and copied here, rounded to 3 decimals. Black numbers are formulas.", f_note)
cell(st, "A4", "Δt: time between letters", f_b); cell(st, "B4", 1, f_in, fmt="0.00", border=True)
cell(st, "C4", "Trained with Δt = 1. Try 2 (letters spaced further apart in time) and watch memory fade faster.", f_note)
cell(st, "A5", "h: size of each thinking step", f_b); cell(st, "B5", 0.5, f_in, fmt="0.00", border=True)
cell(st, "C5", "2 steps × 0.5 = thinking depth 1.0 (as trained). Try 0.25 (thinks only halfway) or 0.75 (overshoots).", f_note)

# Vocabulary + embeddings rows 10-19
r0 = 9
cell(st, f"A{r0-1}", "1. Vocabulary and starting meanings (embeddings): z(t, s=0)", f_b)
for j, t in enumerate(["Character", "Shown as", "Index", "ch 1", "ch 2", "ch 3", "ch 4"]):
    pass
header(st, r0, 1, "Shown as"); header(st, r0, 2, "Character"); header(st, r0, 3, "Index")
for k in range(D): header(st, r0, 4 + k, f"ch {k+1}")
for v in range(V):
    r = r0 + 1 + v
    cell(st, f"A{r}", SHOWN[v], f_n, border=True)
    cell(st, f"B{r}", w["vocab"][v], f_n, border=True, align=Alignment(horizontal="center"))
    cell(st, f"C{r}", v + 1, f_n, border=True, align=Alignment(horizontal="center"))
    for k in range(D):
        cell(st, f"{L(4+k)}{r}", w["emb"][v][k], f_in, fmt=NUM, border=True)
cell(st, "I10", "Each letter starts as 4 numbers (its 'meaning' in 4 channels). "
     "Don't change the characters: the model only knows these 10.", f_note)

# Memory rows 23-26
cell(st, "A22", "2. Memory speeds (the integral)", f_b)
for j, t in enumerate(["Channel", "λ: decay rate", "a = e^(−λ·Δt)", "Half-life (letters)"]):
    header(st, 23 - 0, 1 + j, t) if False else None
header(st, 23, 1, "Channel"); header(st, 23, 2, "λ (learned)"); header(st, 23, 3, "a = e^(−λΔt)"); header(st, 23, 4, "Half-life")
for k in range(D):
    r = 24 + k
    cell(st, f"A{r}", f"ch {k+1}", f_n, border=True)
    cell(st, f"B{r}", w["lam"][k], f_in, fmt=NUM, border=True)
    cell(st, f"C{r}", f"=EXP(-B{r}*$B$4)", f_n, fmt=NUM, border=True)
    cell(st, f"D{r}", f"=LN(2)/(B{r}*$B$4)", f_n, fmt="0.0", border=True)
cell(st, "F24", "a = how much of the old memory survives each letter. Half-life = letters until a memory "
     "has faded to 50%. Small λ means long memory.", f_note)
A_REF = [f"Settings!$C${24+k}" for k in range(D)]

# Vector field rows 31-34
cell(st, "A29", "3. Vector field weights: F = tanh(W · [P, I, V, A] + b)", f_b)
labels = [f"P{k+1}" for k in range(D)] + [f"I{k+1}" for k in range(D)] + [f"V{k+1}" for k in range(D)] + [f"A{k+1}" for k in range(D)]
gcol = ["P"] * 4 + ["I"] * 4 + ["V"] * 4 + ["A"] * 4
header(st, 30, 1, "Output channel")
for j, lab in enumerate(labels): header(st, 30, 2 + j, lab, fill=GROUP[gcol[j]])
header(st, 30, 18, "bias b")
for j in range(D):
    r = 31 + j
    cell(st, f"A{r}", f"rate of change of ch {j+1}", f_n, border=True)
    for i in range(4 * D):
        cell(st, f"{L(2+i)}{r}", w["W"][j][i], f_in, fmt=NUM, border=True)
    cell(st, f"R{r}", w["b"][j], f_in, fmt=NUM, border=True)
cell(st, "A35", "P = present, I = integral (memory), V = velocity (1st derivative), A = acceleration (2nd derivative). "
     "Each row says how much each signal pushes that channel to change.", f_note)
W_ROW = [f"Settings!$B${31+j}:$Q${31+j}" for j in range(D)]
B_REF = [f"Settings!$R${31+j}" for j in range(D)]

# Output rows 39-48
cell(st, "A37", "4. Read-out weights: score(letter) = w · z(t, s=1) + bias", f_b)
header(st, 38, 1, "Letter")
for k in range(D): header(st, 38, 2 + k, f"w ch {k+1}")
header(st, 38, 6, "bias")
for v in range(V):
    r = 39 + v
    cell(st, f"A{r}", f"=A{10+v}", f_n, border=True)
    for k in range(D): cell(st, f"{L(2+k)}{r}", w["Wout"][v][k], f_in, fmt=NUM, border=True)
    cell(st, f"F{r}", w["bout"][v], f_in, fmt=NUM, border=True)
WO_ROW = [f"Settings!$B${39+v}:$E${39+v}" for v in range(V)]
BO_REF = [f"Settings!$F${39+v}" for v in range(V)]
cell(st, "A50", "Parameter count", f_b)
cell(st, "B50", "=COUNT(D10:G19)+COUNT(B24:B27)+COUNT(B31:R34)+COUNT(B39:F48)", f_b, fmt="0", border=True)
cell(st, "C50", "learned numbers in total (the full Python model has 641,601).", f_note)
st.freeze_panes = "B4"

# =========================== Input ===========================
ip = wb.create_sheet("Input")
cell(ip, "A1", "Step 0: Turn text into numbers", f_title)
cell(ip, "A3", "Type 16 letters here →", f_b)
cell(ip, "B3", "the cat sat on t", Font(name=ARIAL, size=14, bold=True, color=BLUE), fill_yellow, border=True)
ip.merge_cells("B3:H3")
cell(ip, "B4", '=IF(LEN(B3)=16,"✓ 16 letters","Note: text has "&LEN(B3)&" letters; it is padded with spaces or cut to 16.")', f_note)
cell(ip, "A5", "Allowed letters: a c e h m n o s t and space. The model learned sentences like "
     "'the cat sat on the mat' and 'the hen ate oats'.", f_note)
hdr = ["Position t", "Character", "Shown as", "Vocab index", "z ch 1", "z ch 2", "z ch 3", "z ch 4",
       "Next letter (actual)", "Next index", "Warning"]
widths = [10, 10, 10, 10, 10, 10, 10, 10, 14, 10, 34]
for j, (h_, wd) in enumerate(zip(hdr, widths)): header(ip, 6, 1 + j, h_, width=wd)
for t in range(1, T + 1):
    r = 6 + t
    cell(ip, f"A{r}", t, f_n, border=True, align=Alignment(horizontal="center"))
    cell(ip, f"B{r}", f'=MID($B$3&REPT(" ",16),A{r},1)', f_n, border=True, align=Alignment(horizontal="center"))
    cell(ip, f"D{r}", f"=IFERROR(MATCH(B{r},Settings!$B$10:$B$19,0),1)", f_n, border=True, align=Alignment(horizontal="center"))
    cell(ip, f"C{r}", f"=INDEX(Settings!$A$10:$A$19,D{r})", f_link, border=True, align=Alignment(horizontal="center"))
    for k in range(D):
        cell(ip, f"{L(5+k)}{r}", f"=INDEX(Settings!$D$10:$G$19,$D{r},{k+1})", f_link, fmt=NUM, border=True, fill=GROUP["P"])
    if t < T:
        cell(ip, f"I{r}", f"=C{r+1}", f_n, border=True, align=Alignment(horizontal="center"))
        cell(ip, f"J{r}", f"=D{r+1}", f_n, border=True, align=Alignment(horizontal="center"))
    else:
        cell(ip, f"I{r}", "? (to predict)", f_note, border=True, align=Alignment(horizontal="center"))
        cell(ip, f"J{r}", "", f_n, border=True)
    cell(ip, f"K{r}", f'=IF(ISNA(MATCH(B{r},Settings!$B$10:$B$19,0)),"Not in vocabulary: treated as a space","")', f_note, border=True)
ip.freeze_panes = "A7"


# =========================== Step sheets ===========================
def step_sheet(name, s_from, s_to, src_sheet, src_cols):
    ws = wb.create_sheet(name)
    cell(ws, "A1", f"{name}: think from depth s = {s_from} to s = {s_to}", f_title)
    cell(ws, "A2", "Each row is one moment in the text. For each moment we measure the present (P), the memory "
         "of the past (I), how fast meaning is changing (V) and whether that change is speeding up (A). "
         "The vector field F turns these into a rate of change, and an Euler step moves z forward: z_new = z + h·F.", f_note)
    ws.column_dimensions["A"].width = 9; ws.column_dimensions["B"].width = 9
    for c in range(3, 27): ws.column_dimensions[L(c)].width = 9
    groups = [("P = z (present)", "P", 3), ("I = integral memory", "I", 7), ("V = velocity dz/dt", "V", 11),
              ("A = acceleration d²z/dt²", "A", 15), ("F = rate of change ∂z/∂s", "F", 19), ("z_new = z + h·F", "Z", 23)]
    for text, g, c1 in groups: group(ws, 5, c1, c1 + 3, text, GROUP[g])
    header(ws, 6, 1, "t"); header(ws, 6, 2, "Letter")
    for text, g, c1 in groups:
        for k in range(D): header(ws, 6, c1 + k, f"{g}{k+1}" if g != "Z" else f"z{k+1}", fill=GROUP[g])
    dt = "Settings!$B$4"
    for t in range(1, T + 1):
        r = 6 + t
        cell(ws, f"A{r}", t, f_n, border=True, align=Alignment(horizontal="center"))
        cell(ws, f"B{r}", f"=Input!C{r}", f_link, border=True, align=Alignment(horizontal="center"))
        for k in range(D):
            P, I, Vc, Ac, Fc, Z = (L(c1 + k) for c1 in (3, 7, 11, 15, 19, 23))
            cell(ws, f"{P}{r}", f"={src_sheet}!{src_cols[k]}{r}", f_link, fmt=NUM, border=True)
            if t == 1:
                fI = f"=(1-{A_REF[k]})*{P}{r}"
                fV = f"=({P}{r}-{P}{r})/{dt}"
                fA = f"=({P}{r}-2*{P}{r}+{P}{r})/{dt}^2"
            elif t == 2:
                fI = f"={A_REF[k]}*{I}{r-1}+(1-{A_REF[k]})*{P}{r}"
                fV = f"=({P}{r}-{P}{r-1})/{dt}"
                fA = f"=({P}{r}-2*{P}{r-1}+{P}{r-1})/{dt}^2"
            else:
                fI = f"={A_REF[k]}*{I}{r-1}+(1-{A_REF[k]})*{P}{r}"
                fV = f"=({P}{r}-{P}{r-1})/{dt}"
                fA = f"=({P}{r}-2*{P}{r-1}+{P}{r-2})/{dt}^2"
            cell(ws, f"{I}{r}", fI, fmt=NUM, border=True)
            cell(ws, f"{Vc}{r}", fV, fmt=NUM, border=True)
            cell(ws, f"{Ac}{r}", fA, fmt=NUM, border=True)
            cell(ws, f"{Fc}{r}", f"=TANH(SUMPRODUCT({W_ROW[k]},$C{r}:$R{r})+{B_REF[k]})", fmt=NUM, border=True)
            cell(ws, f"{Z}{r}", f"={P}{r}+Settings!$B$5*{Fc}{r}", fmt=NUM, border=True, fill=GROUP["Z"])
    cell(ws, "A24", "Rows 1–2 note: there is no letter before position 1, so the model treats the 'previous' "
         "value as equal to the current one, which makes velocity and acceleration start at 0. "
         "Every formula only looks at the current row and rows ABOVE it, so the model never peeks at the future.", f_note)
    ws.freeze_panes = "C7"
    return ws


step_sheet("Step 1", 0, 0.5, "Input", ["E", "F", "G", "H"])
step_sheet("Step 2", 0.5, 1, "'Step 1'", ["W", "X", "Y", "Z"])

# =========================== Prediction ===========================
pr = wb.create_sheet("Prediction")
cell(pr, "A1", "Step 3: From meaning to a prediction", f_title)
cell(pr, "A2", "Scores: each letter gets a score from the final z. Probabilities: softmax turns scores into chances "
     "that add up to 100%. The highest chance is the model's guess. Green = likely, white = unlikely.", f_note)
group(pr, 5, 3, 6, "z at depth s = 1", GROUP["Z"])
group(pr, 5, 7, 16, "Score for each possible next letter", GROUP["S"])
group(pr, 5, 17, 26, "Probability of each next letter (softmax)", GROUP["Pr"])
group(pr, 5, 27, 31, "Result", GROUP["R"])
header(pr, 6, 1, "t", width=6); header(pr, 6, 2, "Letter", width=8)
for k in range(D): header(pr, 6, 3 + k, f"z{k+1}", fill=GROUP["Z"], width=8)
for v in range(V):
    header(pr, 6, 7 + v, SHOWN[v], fill=GROUP["S"], width=8)
    header(pr, 6, 17 + v, SHOWN[v], fill=GROUP["Pr"], width=8)
for j, (t_, wd) in enumerate([("Predicted next", 11), ("Actual next", 10), ("Chance given to actual", 12),
                              ("Right?", 8), ("Surprise (bits)", 10)]):
    header(pr, 6, 27 + j, t_, fill=GROUP["R"], width=wd)
for t in range(1, T + 1):
    r = 6 + t
    cell(pr, f"A{r}", t, f_n, border=True, align=Alignment(horizontal="center"))
    cell(pr, f"B{r}", f"=Input!C{r}", f_link, border=True, align=Alignment(horizontal="center"))
    for k in range(D):
        cell(pr, f"{L(3+k)}{r}", f"='Step 2'!{L(23+k)}{r}", f_link, fmt=NUM, border=True)
    for v in range(V):
        cell(pr, f"{L(7+v)}{r}", f"=SUMPRODUCT({WO_ROW[v]},$C{r}:$F{r})+{BO_REF[v]}", fmt="0.00", border=True)
        cell(pr, f"{L(17+v)}{r}", f"=EXP({L(7+v)}{r}-MAX($G{r}:$P{r}))/SUMPRODUCT(EXP($G{r}:$P{r}-MAX($G{r}:$P{r})))",
             fmt="0%", border=True)
    cell(pr, f"AA{r}", f"=INDEX(Settings!$A$10:$A$19,MATCH(MAX($Q{r}:$Z{r}),$Q{r}:$Z{r},0))", f_b, border=True, align=Alignment(horizontal="center"))
    cell(pr, f"AB{r}", f"=Input!I{r}", f_link, border=True, align=Alignment(horizontal="center"))
    if t < T:
        cell(pr, f"AC{r}", f"=INDEX($Q{r}:$Z{r},Input!J{r})", fmt="0%", border=True)
        cell(pr, f"AD{r}", f'=IF(AA{r}=AB{r},"✓","✗")', border=True, align=Alignment(horizontal="center"))
        cell(pr, f"AE{r}", f"=-LOG(AC{r},2)", fmt="0.00", border=True)
    else:
        for c in ("AC", "AE"): cell(pr, f"{c}{r}", "", border=True)
        cell(pr, f"AD{r}", "guess", f_note, border=True, align=Alignment(horizontal="center"))
pr.conditional_formatting.add("Q7:Z22", ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF",
                                                       end_type="num", end_value=1, end_color="63BE7B"))
cell(pr, "A25", "Summary", f_b)
cell(pr, "A26", "Correct guesses (positions 1–15)"); cell(pr, "E26", '=COUNTIF(AD7:AD21,"✓")/15', f_b, fmt="0%", border=True)
cell(pr, "A27", "Average surprise (bits per letter, lower is better)"); cell(pr, "E27", "=AVERAGE(AE7:AE21)", f_b, fmt="0.00", border=True)
cell(pr, "A28", "Model's guess for letter 17"); cell(pr, "E28", "=AA22", f_b, border=True, fill=fill_yellow)
cell(pr, "A29", "…with confidence"); cell(pr, "E29", "=MAX(Q22:Z22)", f_b, fmt="0%", border=True)
pr.freeze_panes = "C7"

# =========================== How It Works ===========================
hw = wb.create_sheet("How It Works")
for col, wd in zip("ABCDEF", [5, 22, 38, 46, 22, 60]): hw.column_dimensions[col].width = wd
cell(hw, "A1", "How It Works: every formula, how it's used, and why it works", f_title)
cell(hw, "A2", "The model is one equation: ∂z/∂s = F(present, integral of past, 1st derivative, 2nd derivative). "
     "t = position in the text (time). s = thinking depth from 0 to 1. z = the model's understanding (4 numbers per position).", f_note)
hw.merge_cells("A2:F2")
for j, t_ in enumerate(["#", "Stage", "Calculus idea", "Excel formula (example cell)", "Where", "Why it works"]):
    header(hw, 4, 1 + j, t_)
rows = [
    ("Pick the letter", "Read one letter from your text.",
     '=MID($B$3&REPT(" ",16),A7,1)', "Input!B7",
     "Splits your text into 16 moments in time, one letter each. Spaces are added if the text is short."),
    ("Letter → number", "None yet: a lookup.",
     "=IFERROR(MATCH(B7,Settings!$B$10:$B$19,0),1)", "Input!D7",
     "Finds the letter's row in the vocabulary. Unknown letters are treated as a space."),
    ("Starting meaning z(t, 0)", "Initial condition: every differential equation needs a starting point.",
     "=INDEX(Settings!$D$10:$G$19,$D7,1)", "Input!E7:H22",
     "Each letter starts as 4 learned numbers. This is where the 'ball' starts rolling from."),
    ("Memory decay factor", "Exact solution of dm/dt = −λ(m − z): over time Δt, old memory shrinks by e^(−λΔt).",
     "=EXP(-B24*$B$4)", "Settings!C24:C27",
     "Exponential decay is what any 'leaky' system does (cooling coffee, a fading echo). λ was learned, "
     "so the model chose its own memory speeds. For these short words it chose quick-fading memories (half-lives of 1–2 letters)."),
    ("Integral memory I", "Integral: I(t) = ∫ λe^(−λ(t−τ)) z(τ) dτ, a weighted sum of the whole past, recent letters counting most.",
     "=Settings!$C$24*G7+(1-Settings!$C$24)*C8", "Step 1!G7:J22",
     "New memory = (part of old memory that survives) + (share of the new letter). Applied row after row, this adds up "
     "the entire past with fading weights. That is exactly what the integral means, solved exactly, step by step."),
    ("Velocity V", "First derivative dz/dt ≈ (z(t) − z(t−Δt)) / Δt.",
     "=(C8-C7)/Settings!$B$4", "Step 1!K7:N22",
     "A derivative is 'change ÷ time'. Using the PREVIOUS row (a backward difference) means the model "
     "only uses the past and never cheats by looking ahead."),
    ("Acceleration A", "Second derivative d²z/dt² ≈ (z(t) − 2z(t−1) + z(t−2)) / Δt².",
     "=(C9-2*C8+C7)/Settings!$B$4^2", "Step 1!O7:R22",
     "Measures whether the change is speeding up or slowing down, for example when a word is starting versus ending."),
    ("Vector field F", "The rate of change of understanding: ∂z/∂s = F(P, I, V, A).",
     "=TANH(SUMPRODUCT(Settings!$B$31:$Q$31,$C7:$R7)+Settings!$R$31)", "Step 1!S7:V22",
     "SUMPRODUCT weighs the 16 signals (present, memory, velocity, acceleration) by learned weights. TANH keeps "
     "the result between −1 and +1, so the rate of change can never explode."),
    ("Euler step", "Solving the ODE: from the definition of the derivative, z(s+h) ≈ z(s) + h·dz/ds.",
     "=C7+Settings!$B$5*S7", "Step 1!W7:Z22",
     "'New position = old position + speed × time step.' It's how a car's GPS estimates where you'll be in a second. "
     "Smaller steps give a more accurate path. The Python model uses RK4, a more precise version of the same idea."),
    ("Second step", "Keep solving: re-measure P, I, V, A at the new z, then step again.",
     "='Step 1'!W7", "Step 2 (all)",
     "After the first step the understanding has changed, so the memory and the derivatives change too. "
     "Two steps of h = 0.5 reach depth s = 1. More steps means more careful 'thinking'."),
    ("Scores", "None: a weighted sum.",
     "=SUMPRODUCT(Settings!$B$39:$E$39,$C7:$F7)+Settings!$F$39", "Prediction!G7:P22",
     "Each possible next letter has a learned pattern. The score measures how well the final understanding matches it."),
    ("Probabilities (softmax)", "Exponential function e^x.",
     "=EXP(G7-MAX($G7:$P7))/SUMPRODUCT(EXP($G7:$P7-MAX($G7:$P7)))", "Prediction!Q7:Z22",
     "e^score makes every score positive, and dividing by the total makes them add up to 100%. Subtracting the max "
     "first is a standard trick that avoids huge numbers without changing the answer."),
    ("Guess", "None: pick the largest.",
     "=INDEX(Settings!$A$10:$A$19,MATCH(MAX($Q7:$Z7),$Q7:$Z7,0))", "Prediction!AA7:AA22",
     "The letter with the highest probability is the model's prediction."),
    ("Surprise (loss)", "Logarithm: bits = −log₂(chance given to the right answer).",
     "=-LOG(AC7,2)", "Prediction!AE7:AE21",
     "If the model gave the right letter 100%, surprise = 0. At 50%, surprise = 1 bit. Training made this as "
     "small as possible."),
    ("Training (done in Python)", "Gradients: the chain rule from calculus finds, for every weight, which direction lowers the surprise.",
     "(not in Excel)", "excel/train_mini.py",
     "3,000 times: compute the surprise, use derivatives to nudge all 162 weights a tiny bit downhill. "
     "The final weights were copied into the Settings sheet. Excel recreates the 'thinking', not the learning."),
]
for i, (stage, idea, formula, where, why) in enumerate(rows):
    r = 5 + i
    for j, val in enumerate([i + 1, stage, idea, formula, where, why]):
        c = hw.cell(row=r, column=1 + j, value=val)
        c.font = Font(name="Consolas" if j == 3 else ARIAL, size=10, bold=(j == 1))
        c.alignment = wrap; c.border = box
        if j == 3: c.data_type = "s"; c.value = "'" + val if False else val
    hw.row_dimensions[r].height = 75
# formulas shown as text: prefix with a space-safe marker so Excel doesn't evaluate
for i in range(len(rows)):
    c = hw.cell(row=5 + i, column=4)
    c.value = rows[i][2]
    c.data_type = "s"
r = 5 + len(rows) + 1
cell(hw, f"A{r}", "Why does it work?", Font(name=ARIAL, size=13, bold=True, color="1F3864"))
why = [
    "1. Language is a flow. Meaning builds up, shifts and settles as you read. Describing it with rates of change "
    "(derivatives) and accumulated history (integrals) fits that naturally.",
    "2. Like a PID controller (cruise control, thermostats), the model reacts to the Present, the Integral of the past "
    "and the Derivative (trend). Engineers have used these three signals for a century because together they capture "
    "where you are, where you've been and where you're heading.",
    "3. The memory never needs to re-read old letters. It is a running total, so it stays the same size no matter "
    "how long the text is. This is why the full model was faster than a Transformer on long passages.",
    "4. The Euler steps follow the vector field like a ball rolling downhill. Training shaped the 'hills' (the weights) "
    "so that the ball lands on a meaning that predicts the next letter well.",
    "5. Because 'thinking' is a continuous path, you can change the step size h in Settings and see how the "
    "answer changes. A stack of layers can't do this.",
]
for i, t_ in enumerate(why):
    c = hw.cell(row=r + 1 + i, column=1, value=t_); c.font = f_n; c.alignment = wrap
    hw.merge_cells(start_row=r + 1 + i, start_column=1, end_row=r + 1 + i, end_column=6)
    hw.row_dimensions[r + 1 + i].height = 32
r = r + len(why) + 2
cell(hw, f"A{r}", "How this mini version differs from the full Python model", Font(name=ARIAL, size=13, bold=True, color="1F3864"))
diffs = ["• 4 channels instead of 128, 10 letters instead of 65, 162 parameters instead of 641,601.",
         "• 2 Euler steps instead of 4 RK4 steps.",
         "• Leaves out the normalisation, the gate and the depth signal, to keep every formula readable.",
         "• Trained on simple made-up sentences ('the cat sat on the mat') instead of Shakespeare."]
for i, t_ in enumerate(diffs):
    c = hw.cell(row=r + 1 + i, column=1, value=t_); c.font = f_n
    hw.merge_cells(start_row=r + 1 + i, start_column=1, end_row=r + 1 + i, end_column=6)
hw.freeze_panes = "A5"

# =========================== Start Here content ===========================
s = ws0
s.column_dimensions["A"].width = 34; s.column_dimensions["B"].width = 90
cell(s, "A1", "The Calculus Language Model, in Excel", f_title)
cell(s, "A2", "A tiny, working language model built from derivatives, integrals and a differential equation. "
     "Every number is a live formula: change the text and watch it think.", f_note)
cell(s, "A4", "Your text", f_b); cell(s, "B4", "=Input!B3", Font(name=ARIAL, size=12, bold=True, color=GREEN), border=True)
cell(s, "A5", "Model's guess for the next letter", f_b); cell(s, "B5", "=Prediction!E28", Font(name=ARIAL, size=12, bold=True, color=GREEN), fill_yellow, border=True)
cell(s, "A6", "Confidence", f_b); cell(s, "B6", "=Prediction!E29", Font(name=ARIAL, size=12, color=GREEN), fmt="0%", border=True, align=Alignment(horizontal="left"))
cell(s, "A7", "Correct guesses in your text", f_b); cell(s, "B7", "=Prediction!E26", Font(name=ARIAL, size=12, color=GREEN), fmt="0%", border=True, align=Alignment(horizontal="left"))
cell(s, "A9", "How to use it", Font(name=ARIAL, size=13, bold=True, color="1F3864"))
steps = ["1. Go to the Input sheet and type 16 characters in the yellow box (letters a c e h m n o s t and spaces).",
         "   Try: 'the hen ate oats', 'a man sat on a s', 'the ant sat on t'.",
         "2. Follow the sheets left to right: Input → Step 1 → Step 2 → Prediction.",
         "3. Read 'How It Works' for every formula, what it does and why it works.",
         "4. Experiment in Settings: change Δt (time between letters), h (step size) or any blue weight."]
for i, t_ in enumerate(steps): cell(s, f"A{10+i}", t_); s.merge_cells(f"A{10+i}:B{10+i}")
cell(s, "A16", "Sheets", Font(name=ARIAL, size=13, bold=True, color="1F3864"))
sheets = [("Input", "Turns your text into numbers: the starting point z(t, s=0)."),
          ("Step 1", "First thinking step (depth 0 → 0.5): memory, derivatives, rate of change, Euler step."),
          ("Step 2", "Second thinking step (depth 0.5 → 1): the same formulas applied to the updated understanding."),
          ("Prediction", "Scores → probabilities → the predicted next letter, plus how surprised the model was."),
          ("How It Works", "Every formula explained: the calculus idea, the Excel formula, and why it works."),
          ("Settings", "Δt, step size and all 162 learned weights.")]
for i, (n, d) in enumerate(sheets):
    cell(s, f"A{17+i}", n, f_b, border=True); cell(s, f"B{17+i}", d, border=True)
cell(s, "A24", "Colour key", Font(name=ARIAL, size=13, bold=True, color="1F3864"))
cell(s, "A25", "Blue text", f_in, border=True); cell(s, "B25", "Inputs and learned weights: you can change these.", border=True)
cell(s, "A26", "Black text", f_n, border=True); cell(s, "B26", "Formulas: calculated automatically.", border=True)
cell(s, "A27", "Green text", f_link, border=True); cell(s, "B27", "Formulas that pull values from another sheet.", border=True)
cell(s, "A28", "Yellow fill", f_n, fill_yellow, border=True); cell(s, "B28", "The cell to type in, and the final answer.", border=True)

for ws in wb.worksheets:
    ws.sheet_view.showGridLines = False
order = ["Start Here", "How It Works", "Input", "Step 1", "Step 2", "Prediction", "Settings"]
wb._sheets = [wb[n] for n in order]
wb.save("Calculus_LM.xlsx")
print("saved", [ws.title for ws in wb.worksheets])
