import os
import time
import uuid
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="NexaLens - Business Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def api_headers() -> dict[str, str]:
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}


async def api_request(method: str, endpoint: str, **kwargs) -> Any:
    async with httpx.AsyncClient(base_url=API_URL, timeout=60.0) as client:
        response = await client.request(method, endpoint, headers=api_headers(), **kwargs)
        if response.status_code == 401:
            st.session_state.access_token = None
            st.session_state.user = None
            st.rerun()
        response.raise_for_status()
        return response.json()


def login(email: str, password: str) -> bool:
    try:
        import asyncio
        result = asyncio.run(api_request("POST", "/auth/login", data={"username": email, "password": password}))
        st.session_state.access_token = result["access_token"]
        st.session_state.user = asyncio.run(api_request("GET", "/auth/me"))
        return True
    except Exception as e:
        st.error(f"Login failed: {e}")
        return False


def logout():
    st.session_state.access_token = None
    st.session_state.user = None
    st.session_state.chat_history = []


def render_login():
    st.title("📊 NexaLens")
    st.caption("LLM-powered business analytics")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if login(email, password):
                st.rerun()


def render_sidebar():
    with st.sidebar:
        st.title("📊 NexaLens")
        if st.session_state.user:
            st.write(f"👤 {st.session_state.user['name']}")
            st.caption(f"Role: {st.session_state.user['role']}")

        st.divider()

        page = st.radio(
            "Navigation",
            ["💬 Chat", "📁 Data Sources", "📈 Query History", "⚙️ Settings"],
            label_visibility="collapsed",
        )

        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()

        return page


def render_chat():
    st.header("💬 Ask Your Data")

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area(
            "Question",
            placeholder="e.g., What was our total revenue last quarter? Show me the refund policy.",
            height=100,
        )
    with col2:
        st.write("")
        st.write("")
        max_results = st.slider("Max Results", 5, 50, 10)
        include_sql = st.checkbox("Include SQL", value=True)
        include_docs = st.checkbox("Include Documents", value=True)

    if st.button("🔍 Ask", type="primary", use_container_width=True, disabled=not question.strip()):
        with st.spinner("Thinking..."):
            try:
                import asyncio
                request = {
                    "question": question,
                    "max_results": max_results,
                    "include_sql": include_sql,
                    "include_sources": include_docs,
                }
                response = asyncio.run(api_request("POST", "/query", json=request))
                st.session_state.chat_history.insert(0, response)
            except Exception as e:
                st.error(f"Query failed: {e}")

    st.divider()

    for i, msg in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**Q:** {msg['question']}")

            tabs = st.tabs(["💡 Answer", "📊 SQL", "📄 Documents", "📋 Details"])

            with tabs[0]:
                st.markdown(msg["answer"])

            with tabs[1]:
                if msg.get("sql_result"):
                    sql_res = msg["sql_result"]
                    st.code(sql_res["sql"], language="sql")
                    if sql_res["rows"]:
                        df = pd.DataFrame(sql_res["rows"])
                        st.dataframe(df, use_container_width=True)
                        if len(df.columns) >= 2:
                            try:
                                fig = px.bar(df, x=df.columns[0], y=df.columns[1])
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                pass
                    st.caption(f"{sql_res['row_count']} rows · {sql_res['execution_time_ms']:.0f}ms")
                else:
                    st.info("No SQL results for this query")

            with tabs[2]:
                if msg.get("document_results"):
                    for doc in msg["document_results"]:
                        with st.expander(f"📄 {doc['title']} (score: {doc['score']:.2f})"):
                            st.markdown(doc["chunk_content"])
                            st.caption(f"Source: {doc['metadata'].get('filename', 'Unknown')}")
                else:
                    st.info("No document results for this query")

            with tabs[3]:
                col1, col2, col3 = st.columns(3)
                col1.metric("Intent", msg["intent"].upper())
                col2.metric("Confidence", f"{msg['confidence']:.0%}")
                col3.metric("Time", f"{msg['processing_time_ms']:.0f}ms")

            st.divider()


def render_data_sources():
    st.header("📁 Data Sources")

    tab1, tab2 = st.tabs(["📋 List", "➕ Add"])

    with tab1:
        try:
            import asyncio
            sources = asyncio.run(api_request("GET", "/data-sources"))

            if not sources:
                st.info("No data sources configured")
            else:
                for source in sources:
                    with st.expander(f"{'🗄️' if source['type'] == 'sql' else '📄'} {source['name']} ({source['type']})"):
                        st.json(source["config"])
                        if source.get("description"):
                            st.caption(source["description"])
        except Exception as e:
            st.error(f"Failed to load sources: {e}")

    with tab2:
        with st.form("new_source"):
            name = st.text_input("Name")
            source_type = st.selectbox("Type", ["sql", "document"])
            config = st.text_area("Config (JSON)", value='{"host": "localhost", "database": "analytics"}', height=150)
            description = st.text_area("Description")

            if st.form_submit_button("Create", type="primary"):
                try:
                    import json
                    import asyncio
                    asyncio.run(api_request(
                        "POST", "/data-sources",
                        data={"name": name, "type": source_type, "config": config, "description": description}
                    ))
                    st.success("Created!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

        st.divider()
        st.subheader("📄 Upload Documents")
        try:
            import asyncio
            sources = asyncio.run(api_request("GET", "/data-sources"))
            doc_sources = [s for s in sources if s["type"] == "document"]
        except Exception:
            doc_sources = []

        if doc_sources:
            source_id = st.selectbox("Target Source", doc_sources, format_func=lambda s: s["name"])
            uploaded = st.file_uploader("Choose file", type=["pdf", "txt", "md", "csv", "xlsx", "docx"])

            if uploaded and st.button("Upload", type="primary"):
                try:
                    import asyncio
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    data = {"source_id": source_id["id"]}
                    asyncio.run(api_request("POST", f"/data-sources/{source_id['id']}/documents", files=files, data=data))
                    st.success("Document uploaded and indexed!")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
        else:
            st.info("Create a document data source first")


def render_history():
    st.header("📈 Query History")

    try:
        import asyncio
        history = asyncio.run(api_request("GET", "/query/history"))

        if not history:
            st.info("No query history yet")
        else:
            for item in history:
                with st.expander(f"{item['question'][:80]}... ({item['intent']})"):
                    st.markdown(f"**Answer:** {item['answer']}")
                    st.caption(f"Confidence: {item['confidence']:.0%} · Time: {item['processing_time_ms']:.0f}ms · {item['created_at']}")
    except Exception as e:
        st.error(f"Failed to load history: {e}")


def render_settings():
    st.header("⚙️ Settings")

    st.subheader("API Configuration")
    st.code(f"API URL: {API_URL}")

    st.subheader("System Health")
    try:
        import asyncio
        health = asyncio.run(api_request("GET", "/health"))
        for check, status in health["checks"].items():
            st.write(f"{'✅' if status else '❌'} {check}: {'Healthy' if status else 'Unhealthy'}")
    except Exception as e:
        st.error(f"Health check failed: {e}")

    st.divider()
    st.subheader("Danger Zone")
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        st.success("Cleared!")


def main():
    if not st.session_state.access_token:
        render_login()
        return

    page = render_sidebar()

    if page == "💬 Chat":
        render_chat()
    elif page == "📁 Data Sources":
        render_data_sources()
    elif page == "📈 Query History":
        render_history()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()