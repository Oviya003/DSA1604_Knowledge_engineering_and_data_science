"""
ExplainAI - Streamlit Application Styles and Theming
Premium SaaS Light Theme: Deep Navy (#14213D), Modern Blue (#3B82F6), Soft Off-White (#F5F7FB).
"""

CUSTOM_CSS = """
<style>
    /* -------------------------------------------------------------------------
       1. Global Reset & Canvas Styling
       ------------------------------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F5F7FB !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
        color: #1E293B !important;
        -webkit-font-smoothing: antialiased;
    }

    [data-testid="stHeader"] {
        background-color: rgba(245, 247, 251, 0.8) !important;
        backdrop-filter: blur(8px);
    }

    /* Keep main container nicely bounded with comfortable padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1440px !important;
    }

    /* -------------------------------------------------------------------------
       2. Sidebar Navigation & Branding
       ------------------------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        box-shadow: 1px 0 3px rgba(0, 0, 0, 0.02) !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Brand Header in Sidebar */
    .brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 16px;
        margin-bottom: 16px;
        border-bottom: 1px solid #E2E8F0;
    }
    .brand-logo {
        width: 42px;
        height: 42px;
        background: linear-gradient(135deg, #14213D 0%, #1E3A8A 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 20px;
        box-shadow: 0 4px 6px -1px rgba(20, 33, 61, 0.2);
    }
    .brand-title-wrap {
        display: flex;
        flex-direction: column;
    }
    .brand-title {
        font-size: 19px;
        font-weight: 800;
        color: #14213D;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .brand-subtitle {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.06em;
        margin-top: 2px;
    }

    /* Sidebar Section Divider / Heading */
    .nav-section-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-top: 18px;
        margin-bottom: 8px;
        padding-left: 4px;
    }

    /* Style the radio items to look like SaaS Navigation Pills */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 4px !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 8px 12px !important;
        border-radius: 8px !important;
        margin: 0 !important;
        border: 1px solid transparent !important;
        transition: all 0.15s ease-in-out !important;
        cursor: pointer !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        color: #475569 !important;
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #F8FAFC !important;
        color: #14213D !important;
        border-color: #E2E8F0 !important;
    }

    /* Hide the default radio circle */
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    /* Active nav item */
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background-color: #EFF6FF !important;
        color: #1E3A8A !important;
        font-weight: 700 !important;
        border: 1px solid #BFDBFE !important;
        box-shadow: 0 1px 2px rgba(59, 130, 246, 0.08) !important;
    }

    /* Sidebar footer */
    .sidebar-footer {
        margin-top: 24px;
        padding: 12px;
        background-color: #F8FAFC;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        font-size: 11px;
        color: #64748B;
        line-height: 1.4;
    }

    /* -------------------------------------------------------------------------
       3. Main Header & Top Navigation Bar
       ------------------------------------------------------------------------- */
    .header-banner {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .header-title {
        font-size: 24px;
        font-weight: 800;
        margin: 0;
        color: #14213D !important;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        font-size: 13.5px;
        color: #64748B !important;
        margin: 0;
        font-weight: 400;
    }
    .header-pills {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin-top: 8px;
        padding-top: 10px;
        border-top: 1px solid #F1F5F9;
        font-size: 12.5px;
    }
    .header-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 4px 10px;
        border-radius: 6px;
        color: #334155;
        font-weight: 500;
    }
    .header-pill strong {
        color: #14213D;
    }

    /* -------------------------------------------------------------------------
       4. Metric Cards & Containers
       ------------------------------------------------------------------------- */
    .metric-card {
        background: #FFFFFF;
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07);
    }
    .metric-title {
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.06em;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 800;
        margin-top: 4px;
        color: #14213D;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .metric-subtitle {
        font-size: 12px;
        color: #64748B;
        margin-top: 4px;
        font-weight: 400;
    }

    /* Premium SaaS Section Card Wrapper */
    .saas-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .saas-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #F1F5F9;
    }
    .saas-card-title {
        font-size: 16px;
        font-weight: 700;
        color: #14213D;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .saas-card-subtitle {
        font-size: 12.5px;
        color: #64748B;
        margin-top: 2px;
    }

    /* -------------------------------------------------------------------------
       5. Badges & Status Indicators
       ------------------------------------------------------------------------- */
    .badge-at-risk {
        background-color: #FEF2F2;
        color: #DC2626;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 11.5px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border: 1px solid #FECACA;
        letter-spacing: 0.03em;
    }
    .badge-safe {
        background-color: #ECFDF5;
        color: #059669;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 11.5px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border: 1px solid #A7F3D0;
        letter-spacing: 0.03em;
    }
    .badge-info {
        background-color: #EFF6FF;
        color: #2563EB;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 11.5px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border: 1px solid #BFDBFE;
    }
    .badge-warning {
        background-color: #FFFBEB;
        color: #D97706;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 11.5px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border: 1px solid #FDE68A;
    }

    /* -------------------------------------------------------------------------
       6. Alert Callouts & Banners
       ------------------------------------------------------------------------- */
    .warning-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #F59E0B;
        padding: 14px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        color: #92400E;
        font-size: 13px;
        line-height: 1.5;
    }
    .critical-box {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 4px solid #DC2626;
        padding: 14px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        color: #991B1B;
        font-size: 13px;
        line-height: 1.5;
    }
    .info-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        padding: 14px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        color: #166534;
        font-size: 13px;
        line-height: 1.5;
    }

    /* Scientific Disclaimer Card */
    .disclaimer-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 12px;
        color: #475569;
        margin-bottom: 18px;
        line-height: 1.55;
    }
    .disclaimer-card strong {
        color: #1E293B;
    }

    /* -------------------------------------------------------------------------
       7. Recommendation & Triage Cards
       ------------------------------------------------------------------------- */
    .rec-card {
        background: #FFFFFF;
        padding: 18px 20px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .rec-card:hover {
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.06);
    }
    .rec-high {
        border-left: 4px solid #DC2626;
    }
    .rec-med {
        border-left: 4px solid #F59E0B;
    }
    .rec-low {
        border-left: 4px solid #10B981;
    }

    /* -------------------------------------------------------------------------
       8. Streamlit Form & Input Controls Polish
       ------------------------------------------------------------------------- */
    /* Primary buttons */
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(180deg, #3B82F6 0%, #2563EB 100%) !important;
        border: 1px solid #1D4ED8 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        box-shadow: 0 1px 2px 0 rgba(37, 99, 235, 0.3) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: #1D4ED8 !important;
        box-shadow: 0 3px 6px 0 rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-1px);
    }

    /* Secondary buttons */
    .stButton > button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #1E293B !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background: #F8FAFC !important;
        border-color: #94A3B8 !important;
    }

    /* Selectbox, text input, number input styling */
    [data-testid="stSelectbox"] > div,
    [data-testid="stTextInput"] > div,
    [data-testid="stNumberInput"] > div {
        border-radius: 8px !important;
    }

    /* Tab styling */
    [data-testid="stTabs"] button {
        font-weight: 600 !important;
        font-size: 13.5px !important;
        color: #64748B !important;
        padding: 10px 18px !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #1E3A8A !important;
        border-bottom-color: #3B82F6 !important;
        border-bottom-width: 2px !important;
    }

    /* Dataframe containers */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* Section Subheadings */
    h3 {
        font-size: 20px !important;
        font-weight: 800 !important;
        color: #14213D !important;
        letter-spacing: -0.02em !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
    h4 {
        font-size: 15px !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        letter-spacing: -0.01em !important;
        margin-top: 1rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* -------------------------------------------------------------------------
       9. Responsive Layout Adjustments
       ------------------------------------------------------------------------- */
    @media (max-width: 1366px) {
        .block-container {
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }
        .metric-value {
            font-size: 21px !important;
        }
        .header-title {
            font-size: 21px !important;
        }
    }
</style>
"""
