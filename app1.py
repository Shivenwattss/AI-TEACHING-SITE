import streamlit as st
import csv
import os
from datetime import datetime

from kmeans import run_kmeans
from DBSCANS import run_dbscan

st.set_page_config(page_title="Clustering Visualizer", layout="wide")
st.title("Clustering Visualizer")

algo = st.sidebar.selectbox(
    "Choose Algorithm",
    ["K-Means", "DBSCAN"]
)

if algo == "K-Means":
    run_kmeans()
else:
    run_dbscan()

# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

st.markdown("---")
st.markdown("### 💬 Feedback")
st.caption(f"How helpful was the **{algo}** visualizer?")

# 5-star rating
rating = st.feedback(
    "stars",
    key=f"rating_{algo}"
)

# Email
email = st.text_input(
    "📧 Your Email ID",
    placeholder="example@gmail.com",
    key=f"email_{algo}"
)

# Written feedback
comment = st.text_area(
    "📝 Your Feedback",
    placeholder="Tell us what you liked or what could be improved...",
    key=f"comment_{algo}"
)

if st.button("📩 Submit Feedback", type="primary"):

    # Validation
    if rating is None:
        st.warning("⭐ Please select a star rating.")

    elif not email:
        st.warning("📧 Please enter your email ID.")

    elif not comment.strip():
        st.warning("📝 Please write some feedback.")

    else:

        try:
            # ================================================================
            # GET EMAIL SETTINGS FROM STREAMLIT SECRETS
            # ================================================================

            sender_email = st.secrets["EMAIL_ADDRESS"]
            sender_password = st.secrets["EMAIL_PASSWORD"]
            receiver_email = st.secrets["RECEIVER_EMAIL"]

            # ================================================================
            # FEEDBACK DATA
            # ================================================================

            stars = rating + 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ================================================================
            # SAVE TO CSV
            # ================================================================

            feedback_path = os.path.join(
                os.path.dirname(__file__),
                "feedback.csv"
            )

            file_exists = os.path.isfile(feedback_path)

            with open(
                feedback_path,
                "a",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow([
                        "timestamp",
                        "algorithm",
                        "stars",
                        "email",
                        "comment"
                    ])

                writer.writerow([
                    timestamp,
                    algo,
                    stars,
                    email,
                    comment
                ])

            # ================================================================
            # CREATE EMAIL
            # ================================================================

            subject = f"New Feedback - {algo} Visualizer ⭐ {stars}/5"

            body = f"""
You received new feedback for your Clustering Visualizer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Algorithm Used:
{algo}

Rating:
{"⭐" * stars} ({stars}/5)

User Email:
{email}

Feedback:
{comment}

Timestamp:
{timestamp}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = subject

            message.attach(MIMEText(body, "plain"))

            # ================================================================
            # SEND EMAIL
            # ================================================================

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)

            st.success("✅ Thank you! Your feedback has been submitted.")

        except Exception as e:
            st.error(
                "❌ Feedback was saved, but the email could not be sent."
            )
            st.error(f"Error: {e}")
