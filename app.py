import streamlit as st
import os
import json
import hashlib
import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import pandas as pd
import pymupdf
from docx import Document
from sentence_transformers import SentenceTransformer, util


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Knowledge Drift Detection",
    page_icon="📊",
    layout="wide"
)

HISTORY_FILE = "history.json"
MONITORING_FILE = "monitoring.json"
ALERTS_FILE = "alerts.json"
REVIEWS_FILE = "reviews.json"
METADATA_FILE = "document_metadata.json"

VERSION_FOLDER = "document_versions"
MONITOR_FOLDER = "monitored_sources"

os.makedirs(VERSION_FOLDER, exist_ok=True)
os.makedirs(MONITOR_FOLDER, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ============================================================
# JSON FUNCTIONS
# ============================================================

def load_json(filename, default):
    if not os.path.exists(filename):
        return default

    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default


def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(uploaded_file):

    file_name = uploaded_file.name
    extension = os.path.splitext(file_name)[1].lower()

    data = uploaded_file.read()

    if extension == ".pdf":

        doc = pymupdf.open(stream=data, filetype="pdf")

        text = ""

        for page in doc:
            text += page.get_text() + "\n"

        doc.close()

        return text

    elif extension == ".docx":

        temp_file = "temp_document.docx"

        with open(temp_file, "wb") as f:
            f.write(data)

        document = Document(temp_file)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        return text

    elif extension in [".txt", ".md"]:

        return data.decode("utf-8", errors="ignore")

    else:

        return ""


# ============================================================
# HASH
# ============================================================

def calculate_hash(data):

    if isinstance(data, str):
        data = data.encode("utf-8")

    return hashlib.sha256(data).hexdigest()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    text = re.sub(r"\s+", " ", text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        s.strip()
        for s in sentences
        if s.strip()
    ]


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(text1, text2):

    if not text1 or not text2:
        return 0

    embeddings = model.encode(
        [text1, text2],
        convert_to_tensor=True
    )

    score = util.cos_sim(
        embeddings[0],
        embeddings[1]
    ).item()

    return round(float(score), 4)


# ============================================================
# WORDING EXPLANATION
# ============================================================

def wording_explanation(old_text, new_text):

    ratio = SequenceMatcher(
        None,
        old_text.lower(),
        new_text.lower()
    ).ratio()

    if ratio >= 0.90:
        return "Very similar wording with minor changes."

    elif ratio >= 0.70:
        return "Some wording has changed while the main information remains similar."

    elif ratio >= 0.50:
        return "Significant wording changes were detected."

    else:
        return "The wording is substantially different."


# ============================================================
# CHANGE EXPLANATION
# ============================================================

def explain_change(old_text, new_text, change_type, similarity):

    if change_type == "ADDED":
        return "This information appears in the new document but was not found in the previous version."

    if change_type == "REMOVED":
        return "This information existed in the previous document but is missing from the new version."

    if change_type == "UPDATED":

        old_numbers = re.findall(r"\d+(?:\.\d+)?", old_text)
        new_numbers = re.findall(r"\d+(?:\.\d+)?", new_text)

        if old_numbers != new_numbers:
            return (
                "The information appears to be updated, "
                "and numerical values or measurable details have changed."
            )

        return wording_explanation(old_text, new_text)

    if change_type == "POSSIBLE CONFLICT":
        return (
            "The new information is semantically different "
            "from the previous information and may represent a conflict."
        )

    return "No significant change detected."


# ============================================================
# CHANGE CLASSIFICATION
# ============================================================

def classify_change(old_text, new_text, similarity, threshold):

    if old_text.strip().lower() == new_text.strip().lower():
        return "UNCHANGED"

    if similarity >= threshold:
        return "UPDATED"

    return "POSSIBLE CONFLICT"


# ============================================================
# CONFIDENCE
# ============================================================

def get_confidence(similarity):

    if similarity >= 0.90:
        return "High"

    elif similarity >= 0.75:
        return "Medium"

    return "Low"


# ============================================================
# SEVERITY
# ============================================================

def get_severity(drift_score):

    if drift_score < 10:
        return "Low"

    elif drift_score < 30:
        return "Medium"

    return "High"


# ============================================================
# HEALTH SCORE
# ============================================================

def calculate_health(changes):

    if not changes:
        return 100

    total_drift = sum(
        change.get("Drift Score", 0)
        for change in changes
    )

    average_drift = total_drift / len(changes)

    health = max(
        0,
        round(100 - average_drift, 2)
    )

    return health


# ============================================================
# DOCUMENT COMPARISON
# ============================================================

def compare_documents(
    old_text,
    new_text,
    threshold
):

    old_sentences = split_sentences(old_text)
    new_sentences = split_sentences(new_text)

    changes = []

    used_new = set()

    # --------------------------------------------------------
    # OLD → NEW MATCHING
    # --------------------------------------------------------

    for old_sentence in old_sentences:

        best_similarity = 0
        best_index = None
        best_sentence = ""

        for i, new_sentence in enumerate(new_sentences):

            if i in used_new:
                continue

            similarity = semantic_similarity(
                old_sentence,
                new_sentence
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_index = i
                best_sentence = new_sentence

        # ----------------------------------------------------
        # REMOVED
        # ----------------------------------------------------

        if best_index is None or best_similarity < threshold:

            drift_score = 100

            changes.append({
                "Type": "REMOVED",
                "Old Information": old_sentence,
                "New Information": "",
                "Similarity": 0,
                "Drift Score": drift_score,
                "Severity": "High",
                "Confidence": "High",
                "Explanation": explain_change(
                    old_sentence,
                    "",
                    "REMOVED",
                    0
                )
            })

            continue

        used_new.add(best_index)

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        change_type = classify_change(
            old_sentence,
            best_sentence,
            best_similarity,
            threshold
        )

        drift_score = round(
            (1 - best_similarity) * 100,
            2
        )

        severity = get_severity(drift_score)

        confidence = get_confidence(
            best_similarity
        )

        if change_type != "UNCHANGED":

            changes.append({

                "Type": change_type,

                "Old Information":
                    old_sentence,

                "New Information":
                    best_sentence,

                "Similarity":
                    best_similarity,

                "Drift Score":
                    drift_score,

                "Severity":
                    severity,

                "Confidence":
                    confidence,

                "Explanation":
                    explain_change(
                        old_sentence,
                        best_sentence,
                        change_type,
                        best_similarity
                    )
            })

    # --------------------------------------------------------
    # ADDED SENTENCES
    # --------------------------------------------------------

    for i, new_sentence in enumerate(new_sentences):

        if i not in used_new:

            changes.append({

                "Type": "ADDED",

                "Old Information": "",

                "New Information":
                    new_sentence,

                "Similarity": 0,

                "Drift Score": 100,

                "Severity": "High",

                "Confidence": "High",

                "Explanation":
                    explain_change(
                        "",
                        new_sentence,
                        "ADDED",
                        0
                    )
            })

    return changes


# ============================================================
# ALERT CREATION
# ============================================================

def create_alert(change):

    alerts = load_json(
        ALERTS_FILE,
        []
    )

    severity = change.get(
        "Severity",
        "Low"
    )

    if severity not in ["Medium", "High"]:
        return

    old_info = change.get(
        "Old Information",
        ""
    )

    new_info = change.get(
        "New Information",
        ""
    )

    # Duplicate suppression

    for alert in alerts:

        if (
            alert.get("Old Information", "") == old_info
            and
            alert.get("New Information", "") == new_info
            and
            alert.get("Status", "") == "Unread"
        ):

            return

    alert = {

        "Type":
            change.get("Type", ""),

        "Severity":
            severity,

        "Status":
            "Unread",

        "Drift Score":
            change.get("Drift Score", 0),

        "Confidence":
            change.get("Confidence", "Low"),

        "Old Information":
            old_info,

        "New Information":
            new_info,

        "Explanation":
            change.get("Explanation", ""),

        "Created At":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "Read": False
    }

    alerts.append(alert)

    save_json(
        ALERTS_FILE,
        alerts
    )


# ============================================================
# HISTORY
# ============================================================

def save_comparison_history(
    old_document,
    new_document,
    changes,
    health
):

    history = load_json(
        HISTORY_FILE,
        []
    )

    total_changes = len(changes)

    average_drift = 0

    if changes:

        average_drift = round(
            sum(
                c.get("Drift Score", 0)
                for c in changes
            ) / len(changes),
            2
        )

    status = "Healthy"

    if health < 70:
        status = "Needs Attention"

    if health < 40:
        status = "Critical"

    history.append({

        "Title":
            new_document,

        "Source":
            new_document,

        "Old Document":
            old_document,

        "New Document":
            new_document,

        "Changes":
            total_changes,

        "Average Drift":
            average_drift,

        "Health Score":
            health,

        "Status":
            status,

        "Date":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    })

    save_json(
        HISTORY_FILE,
        history
    )


# ============================================================
# DOCUMENT VERSION MANAGEMENT
# ============================================================

def save_document_version(
    uploaded_file,
    title,
    source,
    update_date
):

    data = uploaded_file.getvalue()

    file_hash = calculate_hash(data)

    metadata = load_json(
        METADATA_FILE,
        []
    )

    for item in metadata:

        if item.get("File Hash") == file_hash:

            return False, "Duplicate document detected."

    version_number = 1

    for item in metadata:

        if item.get("Title") == title:

            version_number += 1

    extension = os.path.splitext(
        uploaded_file.name
    )[1]

    stored_name = (
        f"{title}_v{version_number}"
        f"{extension}"
    )

    stored_path = os.path.join(
        VERSION_FOLDER,
        stored_name
    )

    with open(
        stored_path,
        "wb"
    ) as f:

        f.write(data)

    metadata.append({

        "Title":
            title,

        "Version":
            version_number,

        "Source":
            source,

        "Update Date":
            update_date,

        "Original File":
            uploaded_file.name,

        "Stored File":
            stored_name,

        "File Hash":
            file_hash,

        "Saved At":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    })

    save_json(
        METADATA_FILE,
        metadata
    )

    return True, (
        f"Version {version_number} saved successfully."
    )


# ============================================================
# MONITORING
# ============================================================

def get_next_check(frequency):

    now = datetime.now()

    if frequency == "Daily":
        return now + timedelta(days=1)

    if frequency == "Weekly":
        return now + timedelta(days=7)

    return now + timedelta(days=30)


def save_monitored_document(
    name,
    frequency,
    uploaded_file
):

    monitoring = load_json(
        MONITORING_FILE,
        []
    )

    data = uploaded_file.getvalue()

    file_hash = calculate_hash(data)

    extension = os.path.splitext(
        uploaded_file.name
    )[1]

    stored_name = (
        f"{name}_current{extension}"
    )

    stored_path = os.path.join(
        MONITOR_FOLDER,
        stored_name
    )

    with open(
        stored_path,
        "wb"
    ) as f:

        f.write(data)

    existing = None

    for item in monitoring:

        if item.get("Name") == name:
            existing = item
            break

    now = datetime.now()

    record = {

        "Name":
            name,

        "Frequency":
            frequency,

        "Enabled":
            True,

        "Last Checked":
            now.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "Next Check":
            get_next_check(
                frequency
            ).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "Stored File":
            stored_name,

        "File Hash":
            file_hash,

        "Created At":
            now.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }

    if existing:

        monitoring[
            monitoring.index(existing)
        ] = record

    else:

        monitoring.append(record)

    save_json(
        MONITORING_FILE,
        monitoring
    )


def find_monitor(name):

    monitoring = load_json(
        MONITORING_FILE,
        []
    )

    for item in monitoring:

        if item.get("Name") == name:
            return item

    return None


# ============================================================
# AUTOMATED SOURCE CHECK
# ============================================================

def automated_source_check(
    monitor_name,
    uploaded_file,
    threshold
):

    monitor = find_monitor(
        monitor_name
    )

    if not monitor:
        return "Monitoring source not found."

    new_data = uploaded_file.getvalue()

    new_hash = calculate_hash(
        new_data
    )

    old_hash = monitor.get(
        "File Hash",
        ""
    )

    if new_hash == old_hash:

        return "No change detected in the source."

    old_file = os.path.join(
        MONITOR_FOLDER,
        monitor.get(
            "Stored File",
            ""
        )
    )

    if not os.path.exists(old_file):

        return "Previous monitored file not found."

    with open(
        old_file,
        "rb"
    ) as f:

        old_data = f.read()

    old_extension = os.path.splitext(
        old_file
    )[1].lower()

    new_extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    class OldFile:

        def __init__(
            self,
            data,
            name
        ):

            self.data = data
            self.name = name

        def read(self):
            return self.data

    class NewFile:

        def __init__(
            self,
            data,
            name
        ):

            self.data = data
            self.name = name

        def read(self):
            return self.data

    old_file_object = OldFile(
        old_data,
        "old" + old_extension
    )

    new_file_object = NewFile(
        new_data,
        uploaded_file.name
    )

    old_text = extract_text(
        old_file_object
    )

    new_text = extract_text(
        new_file_object
    )

    changes = compare_documents(
        old_text,
        new_text,
        threshold
    )

    health = calculate_health(
        changes
    )

    for change in changes:

        create_alert(change)

    extension = os.path.splitext(
        uploaded_file.name
    )[1]

    new_stored_name = (
        f"{monitor_name}_current"
        f"{extension}"
    )

    new_stored_path = os.path.join(
        MONITOR_FOLDER,
        new_stored_name
    )

    with open(
        new_stored_path,
        "wb"
    ) as f:

        f.write(new_data)

    monitoring = load_json(
        MONITORING_FILE,
        []
    )

    for item in monitoring:

        if item.get("Name") == monitor_name:

            item["Stored File"] = (
                new_stored_name
            )

            item["File Hash"] = (
                new_hash
            )

            item["Last Checked"] = (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            item["Next Check"] = (
                get_next_check(
                    item.get(
                        "Frequency",
                        "Weekly"
                    )
                ).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

    save_json(
        MONITORING_FILE,
        monitoring
    )

    return (
        f"Source changed. "
        f"{len(changes)} changes detected."
    )


# ============================================================
# LOAD DATA FOR DASHBOARD
# ============================================================

history = load_json(
    HISTORY_FILE,
    []
)

alerts = load_json(
    ALERTS_FILE,
    []
)

monitoring = load_json(
    MONITORING_FILE,
    []
)

reviews = load_json(
    REVIEWS_FILE,
    []
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Knowledge Drift")

page = st.sidebar.radio(

    "Navigation",

    [
        "Dashboard",
        "Compare Documents",
        "Change Analysis",
        "Review Center",
        "Alerts",
        "Monitoring",
        "Version History",
        "Reports"
    ]
)

st.sidebar.divider()

threshold = st.sidebar.slider(

    "Semantic Matching Threshold",

    min_value=0.50,

    max_value=0.95,

    value=0.80,

    step=0.01
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("📊 Knowledge Drift Detection Dashboard")

    st.write(
        "Monitor document changes, semantic drift, "
        "alerts and knowledge health."
    )

    # --------------------------------------------------------
    # CURRENT DASHBOARD DATA
    # --------------------------------------------------------

    all_changes = []

    for item in history:

        changes = item.get(
            "Change Details",
            []
        )

        if isinstance(changes, list):
            all_changes.extend(changes)

    # If detailed changes aren't stored in history,
    # use summary information where available.

    total_changes = sum(
        item.get("Changes", 0)
        for item in history
    )

    high_count = sum(
        1
        for alert in alerts
        if alert.get("Severity") == "High"
    )

    medium_count = sum(
        1
        for alert in alerts
        if alert.get("Severity") == "Medium"
    )

    low_count = sum(
        1
        for change in all_changes
        if change.get("Severity") == "Low"
    )

    unread_alerts = sum(

        1
        for alert in alerts

        if (
            alert.get("Status") == "Unread"
            or
            not alert.get("Read", False)
        )
    )

    active_monitors = sum(

        1
        for item in monitoring

        if item.get(
            "Enabled",
            True
        )
    )

    # --------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------

    if history:

        latest_health = history[-1].get(
            "Health Score",
            100
        )

    else:

        latest_health = 100

    if latest_health >= 80:

        health_status = "Healthy"

    elif latest_health >= 60:

        health_status = "Needs Attention"

    else:

        health_status = "Critical"

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Knowledge Health",
            f"{latest_health}%"
        )

    with col2:

        st.metric(
            "Total Changes",
            total_changes
        )

    with col3:

        st.metric(
            "High Severity",
            high_count
        )

    with col4:

        st.metric(
            "Unread Alerts",
            unread_alerts
        )

    with col5:

        st.metric(
            "Monitored Sources",
            active_monitors
        )

    st.divider()

    # --------------------------------------------------------
    # HEALTH SECTION
    # --------------------------------------------------------

    st.subheader("❤️ Knowledge Health")

    st.progress(
        int(
            max(
                0,
                min(
                    100,
                    latest_health
                )
            )
        )
    )

    st.write(
        f"Current status: **{health_status}**"
    )

    # --------------------------------------------------------
    # CHANGE TYPE COUNTS
    # --------------------------------------------------------

    st.subheader("🔄 Change Overview")

    change_types = {
        "Updated": 0,
        "Added": 0,
        "Removed": 0,
        "Possible Conflict": 0
    }

    for change in all_changes:

        change_type = change.get(
            "Type",
            ""
        )

        if change_type in change_types:

            change_types[
                change_type
            ] += 1

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Updated",
            change_types["Updated"]
        )

    with c2:
        st.metric(
            "Added",
            change_types["Added"]
        )

    with c3:
        st.metric(
            "Removed",
            change_types["Removed"]
        )

    with c4:
        st.metric(
            "Possible Conflicts",
            change_types["Possible Conflict"]
        )

    # --------------------------------------------------------
    # CHANGE TYPE CHART
    # --------------------------------------------------------

    if sum(change_types.values()) > 0:

        chart_df = pd.DataFrame({

            "Change Type":
                list(change_types.keys()),

            "Count":
                list(change_types.values())

        })

        st.bar_chart(
            chart_df.set_index(
                "Change Type"
            )
        )

    # --------------------------------------------------------
    # SEVERITY CHART
    # --------------------------------------------------------

    st.subheader("⚠️ Severity Overview")

    severity_data = {

        "High":
            high_count,

        "Medium":
            medium_count,

        "Low":
            low_count
    }

    severity_df = pd.DataFrame({

        "Severity":
            list(
                severity_data.keys()
            ),

        "Count":
            list(
                severity_data.values()
            )
    })

    if severity_df["Count"].sum() > 0:

        st.bar_chart(
            severity_df.set_index(
                "Severity"
            )
        )

    # --------------------------------------------------------
    # DRIFT HISTORY
    # --------------------------------------------------------

    st.subheader("📈 Drift History")

    if history:

        drift_rows = []

        for item in history:

            drift_rows.append({

                "Date":
                    item.get(
                        "Date",
                        ""
                    ),

                "Average Drift":
                    item.get(
                        "Average Drift",
                        0
                    )
            })

        drift_df = pd.DataFrame(
            drift_rows
        )

        if not drift_df.empty:

            st.line_chart(
                drift_df.set_index(
                    "Date"
                )
            )

    else:

        st.info(
            "No drift history available yet."
        )

    # --------------------------------------------------------
    # MONITORING OVERVIEW
    # --------------------------------------------------------

    st.subheader("👁️ Monitoring Overview")

    if monitoring:

        monitor_rows = []

        for item in monitoring:

            monitor_rows.append({

                "Source":
                    item.get(
                        "Name",
                        "Unknown"
                    ),

                "Frequency":
                    item.get(
                        "Frequency",
                        "-"
                    ),

                "Enabled":
                    "Yes"
                    if item.get(
                        "Enabled",
                        True
                    )
                    else "No",

                "Last Checked":
                    item.get(
                        "Last Checked",
                        "-"
                    ),

                "Next Check":
                    item.get(
                        "Next Check",
                        "-"
                    )
            })

        st.dataframe(
            pd.DataFrame(
                monitor_rows
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No sources are currently being monitored."
        )

    # --------------------------------------------------------
    # ALERT OVERVIEW
    # --------------------------------------------------------

    st.subheader("🚨 Recent Alerts")

    if alerts:

        recent_alerts = alerts[-5:]

        alert_rows = []

        for alert in reversed(
            recent_alerts
        ):

            alert_rows.append({

                "Type":
                    alert.get(
                        "Type",
                        "-"
                    ),

                "Severity":
                    alert.get(
                        "Severity",
                        "-"
                    ),

                "Status":
                    alert.get(
                        "Status",
                        "Unread"
                    ),

                "Drift Score":
                    alert.get(
                        "Drift Score",
                        0
                    ),

                "Created At":
                    alert.get(
                        "Created At",
                        "-"
                    )
            })

        st.dataframe(

            pd.DataFrame(
                alert_rows
            ),

            use_container_width=True,

            hide_index=True
        )

    else:

        st.success(
            "No alerts available."
        )

    # --------------------------------------------------------
    # RECENT ACTIVITY
    # --------------------------------------------------------

    st.subheader("🕒 Recent Activity")

    if history:

        activity_rows = []

        for item in reversed(
            history[-10:]
        ):

            activity_rows.append({

                "Date":
                    item.get(
                        "Date",
                        "-"
                    ),

                "Document":
                    item.get(
                        "Title",
                        item.get(
                            "New Document",
                            "-"
                        )
                    ),

                "Changes":
                    item.get(
                        "Changes",
                        0
                    ),

                "Health":
                    item.get(
                        "Health Score",
                        100
                    ),

                "Status":
                    item.get(
                        "Status",
                        "-"
                    )
            })

        st.dataframe(

            pd.DataFrame(
                activity_rows
            ),

            use_container_width=True,

            hide_index=True
        )

    else:

        st.info(
            "No recent activity."
        )


# ============================================================
# COMPARE DOCUMENTS
# ============================================================

elif page == "Compare Documents":

    st.title("📄 Compare Documents")

    old_file = st.file_uploader(
        "Upload Previous Version",
        type=["pdf", "docx", "txt", "md"],
        key="old"
    )

    new_file = st.file_uploader(
        "Upload New Version",
        type=["pdf", "docx", "txt", "md"],
        key="new"
    )

    if old_file and new_file:

        if st.button(
            "🔍 Compare Documents"
        ):

            with st.spinner(
                "Analyzing documents..."
            ):

                old_text = extract_text(
                    old_file
                )

                new_text = extract_text(
                    new_file
                )

                changes = compare_documents(
                    old_text,
                    new_text,
                    threshold
                )

                health = calculate_health(
                    changes
                )

                st.session_state[
                    "changes"
                ] = changes

                st.session_state[
                    "health"
                ] = health

                save_comparison_history(
                    old_file.name,
                    new_file.name,
                    changes,
                    health
                )

                for change in changes:

                    create_alert(change)

            st.success(
                "Document comparison completed."
            )

            st.metric(
                "Knowledge Health",
                f"{health}%"
            )

            st.metric(
                "Changes Detected",
                len(changes)
            )

    if (
        "changes"
        in st.session_state
    ):

        changes = st.session_state[
            "changes"
        ]

        if changes:

            st.subheader(
                "Detected Changes"
            )

            for index, change in enumerate(
                changes,
                start=1
            ):

                with st.expander(
                    f"{index}. {change.get('Type', 'Change')}"
                ):

                    st.write(
                        "**Old Information:**"
                    )

                    st.info(
                        change.get(
                            "Old Information",
                            ""
                        )
                    )

                    st.write(
                        "**New Information:**"
                    )

                    st.success(
                        change.get(
                            "New Information",
                            ""
                        )
                    )

                    st.write(
                        "**Similarity:**",
                        change.get(
                            "Similarity",
                            0
                        )
                    )

                    st.write(
                        "**Drift Score:**",
                        change.get(
                            "Drift Score",
                            0
                        )
                    )

                    st.write(
                        "**Severity:**",
                        change.get(
                            "Severity",
                            "-"
                        )
                    )

                    st.write(
                        "**Confidence:**",
                        change.get(
                            "Confidence",
                            "-"
                        )
                    )

                    st.write(
                        "**Explanation:**",
                        change.get(
                            "Explanation",
                            ""
                        )
                    )

        else:

            st.success(
                "No changes detected."
            )

    # --------------------------------------------------------
    # SAVE VERSION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "💾 Save New Document Version"
    )

    title = st.text_input(
        "Document Title"
    )

    source = st.text_input(
        "Source"
    )

    update_date = st.date_input(
        "Update Date"
    )

    version_file = st.file_uploader(
        "Upload Document Version",
        type=[
            "pdf",
            "docx",
            "txt",
            "md"
        ],
        key="version"
    )

    if version_file:

        if st.button(
            "Save New Document Version"
        ):

            success, message = (
                save_document_version(
                    version_file,
                    title,
                    source,
                    str(update_date)
                )
            )

            if success:

                st.success(
                    message
                )

            else:

                st.warning(
                    message
                )


# ============================================================
# CHANGE ANALYSIS
# ============================================================

elif page == "Change Analysis":

    st.title("🔎 Change Analysis")

    changes = st.session_state.get(
        "changes",
        []
    )

    if not changes:

        st.info(
            "Compare two documents first."
        )

    else:

        for index, change in enumerate(
            changes,
            start=1
        ):

            st.subheader(
                f"Change {index}: "
                f"{change.get('Type')}"
            )

            st.write(
                "**Old:**"
            )

            st.write(
                change.get(
                    "Old Information",
                    ""
                )
            )

            st.write(
                "**New:**"
            )

            st.write(
                change.get(
                    "New Information",
                    ""
                )
            )

            st.write(
                "**Explanation:**"
            )

            st.write(
                change.get(
                    "Explanation",
                    ""
                )
            )

            st.write(
                f"Similarity: "
                f"{change.get('Similarity', 0)}"
            )

            st.write(
                f"Drift Score: "
                f"{change.get('Drift Score', 0)}"
            )

            st.divider()


# ============================================================
# REVIEW CENTER
# ============================================================

elif page == "Review Center":

    st.title("👤 Human Review Center")

    changes = st.session_state.get(
        "changes",
        []
    )

    if not changes:

        st.info(
            "No changes available for review."
        )

    else:

        reviews = load_json(
            REVIEWS_FILE,
            []
        )

        for index, change in enumerate(
            changes
        ):

            st.subheader(
                f"Change {index + 1}"
            )

            st.write(
                change.get(
                    "Explanation",
                    ""
                )
            )

            choice = st.radio(

                "Review Decision",

                [
                    "Accept Change",
                    "Reject Change",
                    "Needs Review"
                ],

                key=f"review_{index}"
            )

            if st.button(
                "Save Decision",
                key=f"save_review_{index}"
            ):

                reviews.append({

                    "Change":
                        change,

                    "Decision":
                        choice,

                    "Reviewed At":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                })

                save_json(
                    REVIEWS_FILE,
                    reviews
                )

                st.success(
                    "Review saved."
                )


# ============================================================
# ALERTS
# ============================================================

elif page == "Alerts":

    st.title("🚨 Alerts & Notifications")

    alerts = load_json(
        ALERTS_FILE,
        []
    )

    if not alerts:

        st.success(
            "No alerts available."
        )

    else:

        for index, alert in enumerate(
            alerts
        ):

            severity = alert.get(
                "Severity",
                "Low"
            )

            status = alert.get(
                "Status",
                "Unread"
            )

            with st.expander(
                f"{severity} Alert - "
                f"{alert.get('Type', 'Change')}"
            ):

                st.write(
                    "**Status:**",
                    status
                )

                st.write(
                    "**Drift Score:**",
                    alert.get(
                        "Drift Score",
                        0
                    )
                )

                st.write(
                    "**Confidence:**",
                    alert.get(
                        "Confidence",
                        "-"
                    )
                )

                st.write(
                    "**Old Information:**"
                )

                st.info(
                    alert.get(
                        "Old Information",
                        ""
                    )
                )

                st.write(
                    "**New Information:**"
                )

                st.success(
                    alert.get(
                        "New Information",
                        ""
                    )
                )

                st.write(
                    "**Explanation:**"
                )

                st.write(
                    alert.get(
                        "Explanation",
                        ""
                    )
                )

                if status != "Read":

                    if st.button(
                        "Mark as Read",
                        key=f"read_{index}"
                    ):

                        alerts[index][
                            "Status"
                        ] = "Read"

                        alerts[index][
                            "Read"
                        ] = True

                        save_json(
                            ALERTS_FILE,
                            alerts
                        )

                        st.rerun()

    if alerts:

        if st.button(
            "Mark All as Read"
        ):

            for alert in alerts:

                alert["Status"] = "Read"
                alert["Read"] = True

            save_json(
                ALERTS_FILE,
                alerts
            )

            st.success(
                "All alerts marked as read."
            )

            st.rerun()


# ============================================================
# MONITORING
# ============================================================

elif page == "Monitoring":

    st.title("👁️ Source Monitoring")

    st.subheader(
        "Add Document / Source"
    )

    monitor_name = st.text_input(
        "Source Name"
    )

    frequency = st.selectbox(
        "Monitoring Frequency",
        [
            "Daily",
            "Weekly",
            "Monthly"
        ]
    )

    monitor_file = st.file_uploader(
        "Upload Current Source",
        type=[
            "pdf",
            "docx",
            "txt",
            "md"
        ],
        key="monitor"
    )

    if st.button(
        "Add / Update Monitoring"
    ):

        if monitor_name and monitor_file:

            save_monitored_document(
                monitor_name,
                frequency,
                monitor_file
            )

            st.success(
                "Monitoring source saved."
            )

        else:

            st.warning(
                "Enter source name and upload a file."
            )

    st.divider()

    st.subheader(
        "Monitored Sources"
    )

    monitoring = load_json(
        MONITORING_FILE,
        []
    )

    for index, item in enumerate(
        monitoring
    ):

        name = item.get(
            "Name",
            "Unknown"
        )

        st.write(
            f"### 📄 {name}"
        )

        st.write(
            "Frequency:",
            item.get(
                "Frequency",
                "-"
            )
        )

        st.write(
            "Last Checked:",
            item.get(
                "Last Checked",
                "-"
            )
        )

        st.write(
            "Next Check:",
            item.get(
                "Next Check",
                "-"
            )
        )

        enabled = st.checkbox(
            "Monitoring Enabled",
            value=item.get(
                "Enabled",
                True
            ),
            key=f"enabled_{index}"
        )

        if enabled != item.get(
            "Enabled",
            True
        ):

            item["Enabled"] = enabled

            save_json(
                MONITORING_FILE,
                monitoring
            )

        new_source = st.file_uploader(

            "Upload New Source Version",

            type=[
                "pdf",
                "docx",
                "txt",
                "md"
            ],

            key=f"new_source_{index}"
        )

        if new_source:

            if st.button(
                "🔍 Check Source Now",
                key=f"check_{index}"
            ):

                result = automated_source_check(

                    name,

                    new_source,

                    threshold
                )

                st.success(
                    result
                )

                st.rerun()

        st.divider()


# ============================================================
# VERSION HISTORY
# ============================================================

elif page == "Version History":

    st.title("📚 Document Version History")

    metadata = load_json(
        METADATA_FILE,
        []
    )

    if not metadata:

        st.info(
            "No document versions saved yet."
        )

    else:

        rows = []

        for item in metadata:

            rows.append({

                "Title":
                    item.get(
                        "Title",
                        "-"
                    ),

                "Version":
                    item.get(
                        "Version",
                        "-"
                    ),

                "Source":
                    item.get(
                        "Source",
                        "-"
                    ),

                "Update Date":
                    item.get(
                        "Update Date",
                        "-"
                    ),

                "Original File":
                    item.get(
                        "Original File",
                        "-"
                    ),

                "Saved At":
                    item.get(
                        "Saved At",
                        "-"
                    )
            })

        st.dataframe(

            pd.DataFrame(
                rows
            ),

            use_container_width=True,

            hide_index=True
        )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":

    st.title("📑 Reports")

    changes = st.session_state.get(
        "changes",
        []
    )

    if changes:

        report_df = pd.DataFrame(
            changes
        )

        st.dataframe(
            report_df,
            use_container_width=True
        )

        csv_data = report_df.to_csv(
            index=False
        )

        st.download_button(

            "⬇️ Download CSV Report",

            csv_data,

            file_name="drift_report.csv",

            mime="text/csv"
        )

        txt_report = ""

        for index, change in enumerate(
            changes,
            start=1
        ):

            txt_report += (
                f"\nCHANGE {index}\n"
            )

            txt_report += (
                f"Type: "
                f"{change.get('Type')}\n"
            )

            txt_report += (
                f"Similarity: "
                f"{change.get('Similarity')}\n"
            )

            txt_report += (
                f"Drift Score: "
                f"{change.get('Drift Score')}\n"
            )

            txt_report += (
                f"Severity: "
                f"{change.get('Severity')}\n"
            )

            txt_report += (
                f"Confidence: "
                f"{change.get('Confidence')}\n"
            )

            txt_report += (
                f"Old Information: "
                f"{change.get('Old Information')}\n"
            )

            txt_report += (
                f"New Information: "
                f"{change.get('New Information')}\n"
            )

            txt_report += (
                f"Explanation: "
                f"{change.get('Explanation')}\n"
            )

            txt_report += (
                "\n----------------------\n"
            )

        st.download_button(

            "⬇️ Download TXT Report",

            txt_report,

            file_name="drift_report.txt",

            mime="text/plain"
        )

    else:

        st.info(
            "Run a document comparison first "
            "to generate a report."
        )