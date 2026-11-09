"""
Synthetic document image generator for DocFraud testing pipeline.

Generates payslips, bank statements, and ID proofs.
All data is entirely fictional. No real personal information is used.
Every generated image is watermarked "SYNTHETIC / TEST DATA".
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFilter, ImageFont

fake = Faker()
Faker.seed(42)
random.seed(42)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "synthetic"

# ── colour palette ──────────────────────────────────────────────────────────
BG_COLOUR      = (255, 255, 255)
HEADER_BG      = (30,  60, 120)
HEADER_FG      = (255, 255, 255)
SECTION_BG     = (230, 240, 255)
BORDER_COLOUR  = (180, 200, 230)
TEXT_DARK      = (30,  30,  30)
TEXT_MID       = (80,  80,  80)
WATERMARK_CLR  = (220, 60,  60)
DIVIDER_CLR    = (200, 210, 230)

# ── font helpers ─────────────────────────────────────────────────────────────
def _font(size: int):
    """Return a truetype font, falling back to the PIL default."""
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()

def _font_bold(size: int):
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return _font(size)


# ── data generation ──────────────────────────────────────────────────────────

def _random_salary_components():
    basic   = random.randint(25_000, 120_000)
    hra     = int(basic * random.uniform(0.30, 0.50))
    travel  = random.randint(1_000,  5_000)
    medical = random.randint(1_250,  2_500)
    special = random.randint(0,     10_000)
    gross   = basic + hra + travel + medical + special

    pf          = int(basic * 0.12)
    tax         = int(gross * random.uniform(0.05, 0.20))
    professional = 200
    total_ded   = pf + tax + professional

    net = gross - total_ded
    return {
        "basic_salary":      basic,
        "hra":               hra,
        "travel_allowance":  travel,
        "medical_allowance": medical,
        "special_allowance": special,
        "gross_earnings":    gross,
        "pf_deduction":      pf,
        "income_tax":        tax,
        "professional_tax":  professional,
        "total_deductions":  total_ded,
        "net_salary":        net,
    }


def generate_fake_payslip() -> dict:
    """
    Return a dict with all fictional payslip fields.

    Fields
    ------
    company_name        : str
    company_address     : str
    employee_name       : str
    employee_id         : str
    designation         : str
    department          : str
    pay_period          : str   e.g. "August 2025"
    bank_account        : str   masked
    basic_salary        : int
    hra                 : int
    travel_allowance    : int
    medical_allowance   : int
    special_allowance   : int
    gross_earnings      : int
    pf_deduction        : int
    income_tax          : int
    professional_tax    : int
    total_deductions    : int
    net_salary          : int
    """
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    year       = random.choice([2024, 2025])
    month      = random.choice(months)
    pay_period = f"{month} {year}"

    salary = _random_salary_components()

    acc_digits = "".join([str(random.randint(0, 9)) for _ in range(12)])
    masked_acc = "XXXX-XXXX-" + acc_digits[-4:]

    designations = [
        "Software Engineer", "Senior Analyst", "Team Lead",
        "Product Manager", "DevOps Engineer", "Data Scientist",
        "QA Engineer", "Business Analyst", "HR Manager", "Finance Executive",
    ]
    departments = [
        "Engineering", "Product", "Finance", "Human Resources",
        "Operations", "Marketing", "Sales", "Legal",
    ]

    return {
        "company_name":    fake.company(),
        "company_address": fake.address().replace("\n", ", "),
        "employee_name":   fake.name(),
        "employee_id":     "EMP" + str(random.randint(10000, 99999)),
        "designation":     random.choice(designations),
        "department":      random.choice(departments),
        "pay_period":      pay_period,
        "bank_account":    masked_acc,
        **salary,
    }


# ── image renderer ───────────────────────────────────────────────────────────

def _fmt(n: int) -> str:
    """Format integer as currency string."""
    return f"₹ {n:,}"


def render_payslip_image(data: dict) -> Image.Image:
    """Render one payslip dict to a PIL Image."""
    W, H = 850, 1050
    img  = Image.new("RGB", (W, H), BG_COLOUR)
    draw = ImageDraw.Draw(img)

    f12 = _font(12)
    f13 = _font(13)
    fb14 = _font_bold(14)
    fb20 = _font_bold(20)
    fb11 = _font_bold(11)

    # outer border
    draw.rectangle([4, 4, W - 5, H - 5], outline=BORDER_COLOUR, width=2)

    # ── HEADER ──────────────────────────────────────────────────────────────
    draw.rectangle([4, 4, W - 5, 90], fill=HEADER_BG)
    draw.text((W // 2, 28), data["company_name"],
              font=fb20, fill=HEADER_FG, anchor="mm")
    draw.text((W // 2, 55), data["company_address"],
              font=f12, fill=(200, 215, 255), anchor="mm")
    draw.text((W // 2, 78), f"Pay Slip for the Month of  {data['pay_period']}",
              font=fb14, fill=(255, 230, 100), anchor="mm")

    y = 105

    # ── EMPLOYEE DETAILS ────────────────────────────────────────────────────
    draw.rectangle([20, y, W - 20, y + 24], fill=SECTION_BG)
    draw.text((30, y + 4), "EMPLOYEE DETAILS", font=fb14, fill=HEADER_BG)
    y += 30

    detail_rows = [
        ("Employee Name",  data["employee_name"],
         "Employee ID",    data["employee_id"]),
        ("Designation",    data["designation"],
         "Department",     data["department"]),
        ("Pay Period",     data["pay_period"],
         "Bank Account",   data["bank_account"]),
    ]
    col1_x, col2_x, col3_x, col4_x = 30, 200, 450, 600
    for label1, val1, label2, val2 in detail_rows:
        draw.text((col1_x, y), label1 + ":", font=fb11, fill=TEXT_MID)
        draw.text((col2_x, y), val1,          font=f12,  fill=TEXT_DARK)
        draw.text((col3_x, y), label2 + ":",  font=fb11, fill=TEXT_MID)
        draw.text((col4_x, y), val2,           font=f12,  fill=TEXT_DARK)
        y += 22

    y += 10
    draw.line([20, y, W - 20, y], fill=DIVIDER_CLR, width=1)
    y += 10

    # ── EARNINGS / DEDUCTIONS table ─────────────────────────────────────────
    draw.rectangle([20, y, W - 20, y + 24], fill=SECTION_BG)
    draw.text((30,          y + 4), "EARNINGS",         font=fb14, fill=HEADER_BG)
    draw.text((W // 2 + 20, y + 4), "DEDUCTIONS",       font=fb14, fill=HEADER_BG)
    draw.line([(W // 2 + 10, y), (W // 2 + 10, y + 24)], fill=BORDER_COLOUR, width=1)
    y += 28

    earn_rows = [
        ("Basic Salary",        _fmt(data["basic_salary"])),
        ("House Rent Allowance", _fmt(data["hra"])),
        ("Travel Allowance",    _fmt(data["travel_allowance"])),
        ("Medical Allowance",   _fmt(data["medical_allowance"])),
        ("Special Allowance",   _fmt(data["special_allowance"])),
    ]
    ded_rows = [
        ("Provident Fund (12%)",  _fmt(data["pf_deduction"])),
        ("Income Tax (TDS)",      _fmt(data["income_tax"])),
        ("Professional Tax",      _fmt(data["professional_tax"])),
    ]

    ex, ea = 30,  200          # earnings label / amount
    dx, da = W // 2 + 20, 710  # deductions label / amount
    row_h  = 22

    max_rows = max(len(earn_rows), len(ded_rows))
    for i in range(max_rows):
        bg = (245, 248, 255) if i % 2 == 0 else BG_COLOUR
        draw.rectangle([20, y, W - 20, y + row_h - 2], fill=bg)

        if i < len(earn_rows):
            lbl, amt = earn_rows[i]
            draw.text((ex, y + 3), lbl, font=f13, fill=TEXT_DARK)
            draw.text((ea, y + 3), amt, font=f13, fill=TEXT_DARK)

        draw.line([(W // 2 + 10, y), (W // 2 + 10, y + row_h)],
                  fill=BORDER_COLOUR, width=1)

        if i < len(ded_rows):
            lbl, amt = ded_rows[i]
            draw.text((dx, y + 3), lbl, font=f13, fill=TEXT_DARK)
            draw.text((da, y + 3), amt, font=f13, fill=TEXT_DARK)

        y += row_h

    y += 6

    # ── TOTALS BAR ──────────────────────────────────────────────────────────
    draw.rectangle([20, y, W - 20, y + 28], fill=(210, 225, 250))
    draw.text((ex, y + 6),  "Gross Earnings:",       font=fb14, fill=HEADER_BG)
    draw.text((ea, y + 6),  _fmt(data["gross_earnings"]),
              font=fb14, fill=HEADER_BG)
    draw.line([(W // 2 + 10, y), (W // 2 + 10, y + 28)],
              fill=BORDER_COLOUR, width=1)
    draw.text((dx, y + 6),  "Total Deductions:",     font=fb14, fill=HEADER_BG)
    draw.text((da, y + 6),  _fmt(data["total_deductions"]),
              font=fb14, fill=HEADER_BG)
    y += 35

    # ── NET SALARY ──────────────────────────────────────────────────────────
    draw.rectangle([20, y, W - 20, y + 38], fill=HEADER_BG)
    draw.text((W // 2, y + 10),
              f"NET SALARY:   {_fmt(data['net_salary'])}",
              font=fb20, fill=(255, 230, 100), anchor="mm")
    y += 48

    y += 12
    draw.line([20, y, W - 20, y], fill=DIVIDER_CLR, width=1)
    y += 12

    # ── NOTES ───────────────────────────────────────────────────────────────
    draw.text((30, y), "Notes:", font=fb11, fill=TEXT_MID)
    y += 18
    notes = [
        "1. This is a computer-generated payslip and does not require a physical signature.",
        "2. All figures are in Indian Rupees (INR).",
        "3. For queries, contact hr@company.example.",
    ]
    for note in notes:
        draw.text((30, y), note, font=f12, fill=TEXT_MID)
        y += 18

    # ── WATERMARK (diagonal) ─────────────────────────────────────────────────
    wm_img  = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    wm_draw = ImageDraw.Draw(wm_img)
    wm_font = _font_bold(52)
    wm_text = "SYNTHETIC / TEST DATA"

    wm_draw.text((W // 2, H // 2 - 60), wm_text,
                 font=wm_font, fill=(220, 60, 60, 60), anchor="mm")
    wm_draw.text((W // 2, H // 2 + 40), wm_text,
                 font=wm_font, fill=(220, 60, 60, 60), anchor="mm")

    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, wm_img)
    final    = img_rgba.convert("RGB")

    # ── FOOTER BANNER ────────────────────────────────────────────────────────
    draw2 = ImageDraw.Draw(final)
    draw2.rectangle([4, H - 30, W - 5, H - 5], fill=(200, 50, 50))
    draw2.text((W // 2, H - 18),
               "★  SYNTHETIC / TEST DATA — NOT A REAL DOCUMENT  ★",
               font=fb11, fill=HEADER_FG, anchor="mm")

    return final


# ── batch generator ──────────────────────────────────────────────────────────

def generate_payslip_batch(
    count: int = 5,
    output_dir: Path = OUTPUT_DIR,
    prefix: str = "payslip",
) -> list[Path]:
    """
    Generate `count` synthetic payslip images.

    Parameters
    ----------
    count      : number of images to generate
    output_dir : directory to save images into
    prefix     : filename prefix (default "payslip")

    Returns
    -------
    List of Path objects for the saved images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i in range(1, count + 1):
        data   = generate_fake_payslip()
        img    = render_payslip_image(data)
        fname  = output_dir / f"{prefix}_{i:03d}.png"
        img.save(fname, format="PNG", optimize=False)
        saved.append(fname)
        print(f"  Saved: {fname.name}")

    return saved


# ── shared watermark helper ──────────────────────────────────────────────────

def _apply_watermark_and_footer(img: Image.Image) -> Image.Image:
    """Apply the SYNTHETIC / TEST DATA watermark and red footer to any image."""
    W, H = img.size
    wm_img  = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    wm_draw = ImageDraw.Draw(wm_img)
    wm_font = _font_bold(min(52, W // 16))
    wm_text = "SYNTHETIC / TEST DATA"
    wm_draw.text((W // 2, H // 2 - 60), wm_text,
                 font=wm_font, fill=(220, 60, 60, 60), anchor="mm")
    wm_draw.text((W // 2, H // 2 + 40), wm_text,
                 font=wm_font, fill=(220, 60, 60, 60), anchor="mm")

    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, wm_img)
    final    = img_rgba.convert("RGB")

    draw2 = ImageDraw.Draw(final)
    fb11  = _font_bold(11)
    draw2.rectangle([4, H - 30, W - 5, H - 5], fill=(200, 50, 50))
    draw2.text((W // 2, H - 18),
               "★  SYNTHETIC / TEST DATA — NOT A REAL DOCUMENT  ★",
               font=fb11, fill=(255, 255, 255), anchor="mm")
    return final


# ── bank statement data generation ───────────────────────────────────────────

def generate_fake_bank_statement() -> dict:
    """
    Return a dict with fictional bank statement fields.

    Fields
    ------
    bank_name           : str
    bank_branch         : str
    account_holder      : str
    account_number      : str   masked
    ifsc_code           : str
    statement_period    : str   e.g. "01 Aug 2025 – 31 Aug 2025"
    opening_balance     : int
    transactions        : list[dict]   each has date, description, debit, credit, balance
    closing_balance     : int
    total_debits        : int
    total_credits       : int
    """
    bank_names = [
        "State Bank of India", "HDFC Bank", "ICICI Bank",
        "Axis Bank", "Punjab National Bank", "Bank of Baroda",
        "Kotak Mahindra Bank", "Yes Bank",
    ]
    year  = random.choice([2024, 2025])
    month = random.randint(1, 12)
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year, 12, 31)
    else:
        end = datetime(year, month + 1, 1) - timedelta(days=1)

    period = f"{start.strftime('%d %b %Y')} – {end.strftime('%d %b %Y')}"

    opening = random.randint(10_000, 500_000)
    balance = opening
    txns = []
    num_txns = random.randint(5, 12)

    descriptions_debit = [
        "UPI/NEFT Transfer", "ATM Withdrawal", "Online Purchase",
        "Utility Bill Payment", "Insurance Premium", "EMI Payment",
        "Mobile Recharge", "Rent Transfer",
    ]
    descriptions_credit = [
        "Salary Credit", "NEFT Inward", "UPI Credit",
        "Interest Credit", "Refund", "Cash Deposit",
    ]

    for j in range(num_txns):
        day = random.randint(start.day, end.day)
        txn_date = datetime(year, month, min(day, end.day)).strftime("%d/%m/%Y")
        is_credit = random.random() < 0.35
        if is_credit:
            amt = random.randint(5_000, 150_000)
            balance += amt
            txns.append({
                "date": txn_date,
                "description": random.choice(descriptions_credit),
                "debit": 0,
                "credit": amt,
                "balance": balance,
            })
        else:
            amt = random.randint(500, min(50_000, max(balance - 1000, 1000)))
            balance -= amt
            txns.append({
                "date": txn_date,
                "description": random.choice(descriptions_debit),
                "debit": amt,
                "credit": 0,
                "balance": balance,
            })

    acc_digits = "".join([str(random.randint(0, 9)) for _ in range(14)])
    masked_acc = "XXXX-XXXX-" + acc_digits[-6:]

    ifsc_letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=4))
    ifsc_code = f"{ifsc_letters}0{random.randint(100000, 999999)}"

    return {
        "bank_name":        random.choice(bank_names),
        "bank_branch":      fake.city() + " Branch",
        "account_holder":   fake.name(),
        "account_number":   masked_acc,
        "ifsc_code":        ifsc_code,
        "statement_period": period,
        "opening_balance":  opening,
        "transactions":     txns,
        "closing_balance":  balance,
        "total_debits":     sum(t["debit"] for t in txns),
        "total_credits":    sum(t["credit"] for t in txns),
    }


def render_bank_statement_image(data: dict) -> Image.Image:
    """Render a bank statement dict to a PIL Image."""
    W, H = 850, 1100
    img  = Image.new("RGB", (W, H), BG_COLOUR)
    draw = ImageDraw.Draw(img)

    f11 = _font(11)
    f12 = _font(12)
    fb11 = _font_bold(11)
    fb14 = _font_bold(14)
    fb18 = _font_bold(18)

    draw.rectangle([4, 4, W - 5, H - 5], outline=BORDER_COLOUR, width=2)

    # header
    draw.rectangle([4, 4, W - 5, 80], fill=(20, 80, 50))
    draw.text((W // 2, 25), data["bank_name"],
              font=fb18, fill=HEADER_FG, anchor="mm")
    draw.text((W // 2, 50), f"Branch: {data['bank_branch']}",
              font=f12, fill=(200, 230, 210), anchor="mm")
    draw.text((W // 2, 68), "ACCOUNT STATEMENT",
              font=fb14, fill=(255, 230, 100), anchor="mm")

    y = 95

    # account details
    draw.rectangle([20, y, W - 20, y + 22], fill=SECTION_BG)
    draw.text((30, y + 4), "ACCOUNT DETAILS", font=fb14, fill=(20, 80, 50))
    y += 28

    details = [
        ("Account Holder:", data["account_holder"],
         "Account No:",     data["account_number"]),
        ("IFSC Code:",      data["ifsc_code"],
         "Period:",         data["statement_period"]),
    ]
    for l1, v1, l2, v2 in details:
        draw.text((30, y),  l1, font=fb11, fill=TEXT_MID)
        draw.text((180, y), v1, font=f12,  fill=TEXT_DARK)
        draw.text((450, y), l2, font=fb11, fill=TEXT_MID)
        draw.text((560, y), v2, font=f12,  fill=TEXT_DARK)
        y += 20

    y += 8
    draw.text((30, y), f"Opening Balance: {_fmt(data['opening_balance'])}",
              font=fb11, fill=TEXT_DARK)
    y += 22

    # transaction table header
    draw.rectangle([20, y, W - 20, y + 20], fill=(20, 80, 50))
    headers = ["Date", "Description", "Debit", "Credit", "Balance"]
    col_x = [30, 130, 380, 510, 680]
    for cx, hdr in zip(col_x, headers):
        draw.text((cx, y + 4), hdr, font=fb11, fill=HEADER_FG)
    y += 22

    for i, txn in enumerate(data["transactions"]):
        bg = (240, 248, 240) if i % 2 == 0 else BG_COLOUR
        draw.rectangle([20, y, W - 20, y + 18], fill=bg)
        draw.text((col_x[0], y + 3), txn["date"],        font=f11, fill=TEXT_DARK)
        draw.text((col_x[1], y + 3), txn["description"], font=f11, fill=TEXT_DARK)
        draw.text((col_x[2], y + 3),
                  _fmt(txn["debit"]) if txn["debit"] else "-",
                  font=f11, fill=(180, 40, 40) if txn["debit"] else TEXT_MID)
        draw.text((col_x[3], y + 3),
                  _fmt(txn["credit"]) if txn["credit"] else "-",
                  font=f11, fill=(40, 130, 40) if txn["credit"] else TEXT_MID)
        draw.text((col_x[4], y + 3), _fmt(txn["balance"]), font=f11, fill=TEXT_DARK)
        y += 19

    y += 10
    draw.line([20, y, W - 20, y], fill=DIVIDER_CLR, width=1)
    y += 8

    # summary
    draw.rectangle([20, y, W - 20, y + 50], fill=(220, 235, 225))
    draw.text((30, y + 5),  f"Total Debits:    {_fmt(data['total_debits'])}",
              font=fb14, fill=(180, 40, 40))
    draw.text((30, y + 27), f"Total Credits:   {_fmt(data['total_credits'])}",
              font=fb14, fill=(40, 130, 40))
    draw.text((450, y + 14), f"Closing Balance:  {_fmt(data['closing_balance'])}",
              font=fb14, fill=(20, 80, 50))
    y += 60

    return _apply_watermark_and_footer(img)


# ── ID proof data generation ─────────────────────────────────────────────────

def generate_fake_id_proof() -> dict:
    """
    Return a dict with fictional government ID proof fields.

    Fields
    ------
    id_type         : str   "Aadhaar Card" or "PAN Card"
    holder_name     : str
    id_number       : str   masked
    date_of_birth   : str
    gender          : str
    address         : str
    issue_date      : str
    father_name     : str   (Aadhaar only)
    """
    id_type = random.choice(["Aadhaar Card", "PAN Card"])

    dob_year  = random.randint(1960, 2002)
    dob_month = random.randint(1, 12)
    dob_day   = random.randint(1, 28)
    dob       = f"{dob_day:02d}/{dob_month:02d}/{dob_year}"

    gender = random.choice(["Male", "Female"])

    issue_year  = random.randint(2015, 2025)
    issue_month = random.randint(1, 12)
    issue_date  = f"{random.randint(1,28):02d}/{issue_month:02d}/{issue_year}"

    result = {
        "id_type":       id_type,
        "holder_name":   fake.name(),
        "date_of_birth": dob,
        "gender":        gender,
        "address":       fake.address().replace("\n", ", "),
        "issue_date":    issue_date,
    }

    if id_type == "Aadhaar Card":
        digits = "".join([str(random.randint(0, 9)) for _ in range(12)])
        result["id_number"]   = f"XXXX-XXXX-{digits[-4:]}"
        result["father_name"] = fake.name_male()
    else:
        letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
        digits  = "".join([str(random.randint(0, 9)) for _ in range(4)])
        result["id_number"]   = f"{letters}{digits}X"
        result["father_name"] = ""

    return result


def render_id_proof_image(data: dict) -> Image.Image:
    """Render an ID proof dict to a PIL Image."""
    W, H = 850, 550
    img  = Image.new("RGB", (W, H), (255, 252, 245))
    draw = ImageDraw.Draw(img)

    f12  = _font(12)
    f13  = _font(13)
    fb11 = _font_bold(11)
    fb14 = _font_bold(14)
    fb18 = _font_bold(18)

    draw.rectangle([4, 4, W - 5, H - 5], outline=(180, 160, 120), width=2)

    is_aadhaar = data["id_type"] == "Aadhaar Card"

    if is_aadhaar:
        hdr_bg = (220, 100, 30)
        hdr_title = "GOVERNMENT OF INDIA"
        hdr_sub   = "Unique Identification Authority of India"
        doc_title = "AADHAAR CARD"
    else:
        hdr_bg = (40, 60, 120)
        hdr_title = "INCOME TAX DEPARTMENT"
        hdr_sub   = "Government of India"
        doc_title = "PERMANENT ACCOUNT NUMBER CARD"

    draw.rectangle([4, 4, W - 5, 75], fill=hdr_bg)
    draw.text((W // 2, 22), hdr_title,
              font=fb18, fill=HEADER_FG, anchor="mm")
    draw.text((W // 2, 42), hdr_sub,
              font=f12, fill=(220, 220, 240), anchor="mm")
    draw.text((W // 2, 62), doc_title,
              font=fb14, fill=(255, 230, 100), anchor="mm")

    y = 95

    # photo placeholder
    draw.rectangle([30, y, 170, y + 170], outline=(150, 140, 120), width=2)
    draw.text((100, y + 80), "PHOTO", font=fb14, fill=(180, 170, 150), anchor="mm")

    # details
    dx = 200
    fields = [
        ("Name:",          data["holder_name"]),
        ("ID Number:",     data["id_number"]),
        ("Date of Birth:", data["date_of_birth"]),
        ("Gender:",        data["gender"]),
    ]
    if is_aadhaar and data.get("father_name"):
        fields.append(("Father's Name:", data["father_name"]))
    fields.append(("Address:", data["address"]))

    for label, val in fields:
        draw.text((dx, y), label, font=fb11, fill=TEXT_MID)
        draw.text((dx + 130, y), val, font=f13, fill=TEXT_DARK)
        y += 24

    y += 10
    draw.text((dx, y), f"Issue Date: {data['issue_date']}",
              font=f12, fill=TEXT_MID)

    return _apply_watermark_and_footer(img)


# ── tampering simulation ────────────────────────────────────────────────────

def _apply_tampering(img: Image.Image, method: str) -> Image.Image:
    """
    Apply a visible synthetic tampering effect to an image.

    Methods
    -------
    blur_region    : blur a rectangular region (simulates redaction/hiding)
    color_shift    : shift colour of a rectangular region
    noise_patch    : add random noise to a region
    text_overlay   : overlay fake replacement text on a region
    """
    W, H = img.size
    img = img.copy()

    rx = random.randint(50, W - 250)
    ry = random.randint(80, H - 150)
    rw = random.randint(120, 200)
    rh = random.randint(30, 60)

    if method == "blur_region":
        region = img.crop((rx, ry, rx + rw, ry + rh))
        region = region.filter(ImageFilter.GaussianBlur(radius=6))
        img.paste(region, (rx, ry))

    elif method == "color_shift":
        for x in range(rx, min(rx + rw, W)):
            for y in range(ry, min(ry + rh, H)):
                r, g, b = img.getpixel((x, y))
                img.putpixel((x, y), (
                    min(255, r + 40),
                    max(0, g - 20),
                    max(0, b - 20),
                ))

    elif method == "noise_patch":
        for x in range(rx, min(rx + rw, W)):
            for y in range(ry, min(ry + rh, H)):
                r, g, b = img.getpixel((x, y))
                noise = random.randint(-30, 30)
                img.putpixel((x, y), (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise)),
                    max(0, min(255, b + noise)),
                ))

    elif method == "text_overlay":
        draw = ImageDraw.Draw(img)
        draw.rectangle([rx, ry, rx + rw, ry + rh], fill=(255, 255, 255))
        overlay_font = _font_bold(14)
        draw.text((rx + 5, ry + 5), "ALTERED VALUE",
                  font=overlay_font, fill=(60, 60, 60))

    return img


TAMPER_METHODS = ["blur_region", "color_shift", "noise_patch", "text_overlay"]


# ── Day 2: organised dataset generator ──────────────────────────────────────

def generate_full_dataset(
    output_dir: Path = OUTPUT_DIR,
    payslip_count: int = 10,
    bank_stmt_count: int = 8,
    id_proof_count: int = 7,
    tamper_ratio: float = 0.4,
    start_index: int = 1,
) -> list[dict]:
    """
    Generate a mixed synthetic dataset with genuine and tampered documents.

    Creates subdirectories genuine/ and tampered/ under output_dir,
    generates documents, and returns metadata rows for the CSV index.
    Numbering begins at ``start_index``; an existing file is never overwritten.
    """
    genuine_dir  = Path(output_dir) / "genuine"
    tampered_dir = Path(output_dir) / "tampered"
    genuine_dir.mkdir(parents=True, exist_ok=True)
    tampered_dir.mkdir(parents=True, exist_ok=True)

    doc_specs = []
    for i in range(payslip_count):
        doc_specs.append(("payslip", i + 1))
    for i in range(bank_stmt_count):
        doc_specs.append(("bank_statement", i + 1))
    for i in range(id_proof_count):
        doc_specs.append(("id_proof", i + 1))

    rows = []
    global_idx = start_index - 1
    for doc_type, seq in doc_specs:
        global_idx += 1
        is_tampered = random.random() < tamper_ratio

        if doc_type == "payslip":
            data = generate_fake_payslip()
            img  = render_payslip_image(data)
            prefix = "payslip"
        elif doc_type == "bank_statement":
            data = generate_fake_bank_statement()
            img  = render_bank_statement_image(data)
            prefix = "bank_stmt"
        else:
            data = generate_fake_id_proof()
            img  = render_id_proof_image(data)
            prefix = "id_proof"

        tamper_method = ""
        if is_tampered:
            tamper_method = random.choice(TAMPER_METHODS)
            img = _apply_tampering(img, tamper_method)
            label = "tampered"
            dest_dir = tampered_dir
        else:
            label = "genuine"
            dest_dir = genuine_dir

        fname = f"{prefix}_{global_idx:03d}.png"
        fpath = dest_dir / fname
        for existing in (genuine_dir / fname, tampered_dir / fname):
            if existing.exists():
                raise FileExistsError(f"Refusing to overwrite existing document: {existing}")
        img.save(fpath, format="PNG", optimize=False)

        row = {
            "filename":       fname,
            "path":           str(fpath.relative_to(output_dir)),
            "doc_type":       doc_type,
            "label":          label,
            "tamper_method":  tamper_method,
            "width":          img.size[0],
            "height":         img.size[1],
            "file_size_bytes": fpath.stat().st_size,
        }
        rows.append(row)
        print(f"  [{label:>8}] {row['path']}")

    return rows


def write_index_csv(rows: list[dict], output_dir: Path = OUTPUT_DIR):
    """Write the dataset index CSV."""
    csv_path = Path(output_dir) / "index.csv"
    fieldnames = [
        "filename", "path", "doc_type", "label",
        "tamper_method", "width", "height", "file_size_bytes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Index written: {csv_path}  ({len(rows)} rows)")
    return csv_path


def append_index_csv(rows: list[dict], output_dir: Path = OUTPUT_DIR):
    """Append rows to an existing index CSV, rejecting duplicate filenames."""
    csv_path = Path(output_dir) / "index.csv"
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing = {r["filename"] for r in reader}
    clashes = existing & {r["filename"] for r in rows}
    if clashes:
        raise ValueError(f"Filenames already indexed: {sorted(clashes)}")
    with open(csv_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
    print(f"\n  Index appended: {csv_path}  (+{len(rows)} rows, {len(existing) + len(rows)} total)")
    return csv_path


def _next_index(output_dir: Path = OUTPUT_DIR) -> int:
    with open(Path(output_dir) / "index.csv", newline="") as f:
        nums = [int(Path(r["filename"]).stem.rsplit("_", 1)[1]) for r in csv.DictReader(f)]
    return max(nums, default=0) + 1


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    Faker.seed(42)
    random.seed(42)

    if "--day2" in sys.argv:
        print(f"Day 2: Generating full synthetic dataset → {OUTPUT_DIR}")
        rows = generate_full_dataset(
            output_dir=OUTPUT_DIR,
            payslip_count=10,
            bank_stmt_count=8,
            id_proof_count=7,
        )
        write_index_csv(rows, OUTPUT_DIR)
        genuine  = sum(1 for r in rows if r["label"] == "genuine")
        tampered = sum(1 for r in rows if r["label"] == "tampered")
        print(f"\nDone — {len(rows)} documents ({genuine} genuine, {tampered} tampered)")
    elif "--day4" in sys.argv:
        # Distinct seed so new documents differ from the Day 2 set.
        Faker.seed(4)
        random.seed(4)
        start = _next_index(OUTPUT_DIR)
        print(f"Day 4: Expanding dataset from index {start:03d} → {OUTPUT_DIR}")
        rows = generate_full_dataset(
            output_dir=OUTPUT_DIR,
            payslip_count=20,
            bank_stmt_count=15,
            id_proof_count=15,
            start_index=start,
        )
        append_index_csv(rows, OUTPUT_DIR)
        genuine  = sum(1 for r in rows if r["label"] == "genuine")
        tampered = sum(1 for r in rows if r["label"] == "tampered")
        print(f"\nDone — {len(rows)} new documents ({genuine} genuine, {tampered} tampered)")
    else:
        print(f"Generating synthetic payslips → {OUTPUT_DIR}")
        paths = generate_payslip_batch(count=5)
        print(f"\nDone — {len(paths)} payslip(s) generated.")
        for p in paths:
            size = p.stat().st_size
            print(f"  {p.name}  ({size:,} bytes)")
