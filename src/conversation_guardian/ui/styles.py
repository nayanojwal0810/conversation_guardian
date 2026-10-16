"""
Professional styling and layout CSS for the ConversationGuardian Moderator Dashboard.

Implements an industry-grade analytics aesthetic:
- Clean navy, white, and light-slate foundation
- Restrained teal accents (#0D9488)
- Accessible amber (#D97706) for warnings and red (#DC2626) strictly for alerts
- High-contrast, minimal card styling without decorative clutter
"""

CUSTOM_CSS = """
<style>
/* Global resets & typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1E293B;
}

/* App background */
.stApp {
    background-color: #F8FAFC;
}

/* Header bar */
.cg-header {
    background: #0F172A;
    padding: 1.25rem 2rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.cg-header h1 {
    color: #FFFFFF;
    font-size: 1.35rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}

.cg-header p {
    color: #94A3B8;
    font-size: 0.85rem;
    margin: 0.2rem 0 0 0;
}

.cg-header-status {
    background: #1E293B;
    border: 1px solid #334155;
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    color: #38BDF8;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

/* KPI Cards */
.cg-kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1.25rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    height: 100%;
}

.cg-kpi-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 0.4rem;
}

.cg-kpi-val {
    font-size: 2rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}

.cg-kpi-sub {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 0.3rem;
}

/* Alert Banners */
.cg-alert-banner {
    padding: 1rem 1.25rem;
    border-radius: 6px;
    margin: 1rem 0;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.cg-alert-danger {
    background: #FEF2F2;
    border-left: 4px solid #DC2626;
    color: #991B1B;
    font-weight: 600;
}

.cg-alert-safe {
    background: #F0FDF4;
    border-left: 4px solid #0D9488;
    color: #065F46;
    font-weight: 600;
}

.cg-alert-warning {
    background: #FFFBEB;
    border-left: 4px solid #D97706;
    color: #92400E;
}

/* Status Badges */
.cg-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.cg-badge-pending { background: #FEF3C7; color: #92400E; }
.cg-badge-in-review { background: #E0E7FF; color: #3730A3; }
.cg-badge-resolved { background: #DCFCE7; color: #166534; }
.cg-badge-dismissed { background: #F1F5F9; color: #475569; }

.cg-badge-alert-true { background: #FEE2E2; color: #991B1B; }
.cg-badge-alert-false { background: #CCFBF1; color: #115E59; }

/* Details card */
.cg-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.cg-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 1rem;
    border-bottom: 1px solid #F1F5F9;
    padding-bottom: 0.5rem;
}

/* Disclaimer notice */
.cg-disclaimer {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    color: #475569;
    margin-top: 1rem;
    line-height: 1.4;
}

/* Table styling adjustments */
.stDataFrame {
    border: 1px solid #E2E8F0;
    border-radius: 6px;
}

/* Buttons */
.stButton>button {
    background-color: #0F172A;
    color: #FFFFFF;
    font-weight: 600;
    border-radius: 6px;
    border: none;
    padding: 0.5rem 1.25rem;
    transition: all 0.15s ease-in-out;
}

.stButton>button:hover {
    background-color: #1E293B;
    color: #38BDF8;
}

/* Form inputs */
.stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
    border-radius: 6px;
    border: 1px solid #CBD5E1;
}

.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
    border-color: #0D9488;
    box-shadow: 0 0 0 1px #0D9488;
}

</style>
"""
