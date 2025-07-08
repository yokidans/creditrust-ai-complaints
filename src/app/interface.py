import gradio as gr
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from src.core.rag_engine import EliteRAGSystem
from src.services.vector_store import get_vector_store
import logging
import time
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class EliteChatInterface:
    def __init__(self, rag_system: EliteRAGSystem):
        self.rag = rag_system
        self.session_history = []
        self.performance_metrics = {
            "query_count": 0,
            "avg_response_time": 0,
            "total_tokens_processed": 0
        }

    def _format_sources(self, sources: List[Dict]) -> pd.DataFrame:
        """Create interactive source display with enhanced metadata"""
        return pd.DataFrame([{
            'Product': s.get('product', 'N/A'),
            'Date': s.get('date', 'Unknown'),
            'Confidence': f"{s.get('similarity_score', 0):.2f}",
            'Excerpt': (s.get('text', '')[:150] + '...') if s.get('text') else '',
            'Complaint ID': s.get('complaint_id', 'N/A')
        } for s in sources])

    def _stream_response(self, query: str, chat_history: List[Tuple[str, str]]):
        """Generate streaming response with sources"""
        start_time = time.time()
        try:
            # Retrieve and process
            retrieved = self.rag.retrieve(query)
            context = [doc.page_content for doc in retrieved]  # Changed from .text to .page_content
            
            # Generate response
            response = self.rag.generate_response(query, context)
            
            # Update performance metrics
            self._update_metrics(start_time, response)
            
            # Format output
            sources_df = self._format_sources([doc.metadata for doc in retrieved])
            
            # Update session history
            self.session_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'sources': sources_df.to_dict('records'),
                'context': context
            })
            
            # Stream output
            partial_response = ""
            for token in response.split():
                partial_response += token + " "
                yield chat_history + [(query, partial_response)], sources_df
                
        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            error_msg = f"⚠️ Analysis Error: {str(e)}"
            yield chat_history + [(query, error_msg)], pd.DataFrame()

    def _update_metrics(self, start_time: float, response: str):
        """Track performance metrics"""
        duration = time.time() - start_time
        self.performance_metrics["query_count"] += 1
        self.performance_metrics["total_tokens_processed"] += len(response.split())
        self.performance_metrics["avg_response_time"] = (
            self.performance_metrics["avg_response_time"] * (self.performance_metrics["query_count"] - 1) + duration
        ) / self.performance_metrics["query_count"]

    def _get_history_preview(self):
        """Return last 5 history items for preview with timestamps"""
        return [
            [entry['timestamp'], entry['query'], entry['response'][:100] + "..."] 
            for entry in self.session_history[-5:]
        ] if self.session_history else []

    def export_session(self):
        """Export full session data as JSON"""
        export_data = {
            "session_data": self.session_history,
            "performance_metrics": self.performance_metrics
        }
        return json.dumps(export_data, indent=2)

    def _save_export(self, export_data: str):
        """Handle file export"""
        export_path = Path("exports")
        export_path.mkdir(exist_ok=True)
        filename = f"session_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = export_path / filename
        with open(filepath, "w") as f:
            f.write(export_data)
        return str(filepath)

    def launch(self):
        """Launch elite Gradio interface with enhanced features"""
        css = """
        .gradio-container {
            max-width: 1200px !important;
            font-family: 'Segoe UI', sans-serif;
        }
        .chatbot {
            min-height: 500px;
            border-radius: 8px;
        }
        .dataframe {
            font-size: 0.9em;
        }
        .highlight {
            background-color: #f5f5f5;
            border-left: 3px solid #4CAF50;
            padding: 0.5em;
        }
        """

        with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
            # Header with branding
            with gr.Row():
                gr.Markdown("""
                # 🚀 CrediTrust AI Analyst 
                ### *Financial Complaint Analysis System - v2.1*
                """)
            
            with gr.Row():
                # Main Chat Column
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        height=600,
                        label="Complaint Analysis",
                        show_copy_button=True,
                        render_markdown=True,
                        avatar_images=(
                            None,  # User avatar
                            None   # Assistant avatar
                        )
                    )
                    
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Ask about financial complaints",
                            placeholder="E.g. What are emerging risks in credit card complaints?",
                            lines=2,
                            max_lines=5
                        )
                    
                    with gr.Row():
                        submit_btn = gr.Button("Analyze", variant="primary")
                        clear_btn = gr.Button("Clear Session", variant="secondary")
                        export_btn = gr.Button("Export Session", variant="secondary")
                
                # Analytics Column
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 Retrieved Sources")
                    with gr.Row():
                        confidence_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.7,
                            step=0.05,
                            label="Minimum Confidence Threshold"
                        )
                        filter_btn = gr.Button("Filter", size="sm")
                    
                    sources_output = gr.Dataframe(
                        headers=["Product", "Date", "Confidence", "Excerpt"],
                        interactive=False,
                        wrap=True
                    )
                    
                    gr.Markdown("### 📊 Session Metrics")
                    metrics_output = gr.JSON(
                        value={
                            "queries_processed": 0,
                            "avg_response_time": 0,
                            "tokens_processed": 0
                        },
                        label="Performance Metrics"
                    )
                    
                    gr.Markdown("### 📝 Session History")
                    history_output = gr.DataFrame(
                        headers=["Time", "Query", "Response Snippet"],
                        interactive=False
                    )

            # Hidden components
            export_data = gr.Textbox(visible=False)
            export_file = gr.File(visible=False)
            
            # Event handlers
            submit_event = query_input.submit(
                fn=self._stream_response,
                inputs=[query_input, chatbot],
                outputs=[chatbot, sources_output],
                concurrency_limit=3
            )
            
            submit_btn.click(
                fn=self._stream_response,
                inputs=[query_input, chatbot],
                outputs=[chatbot, sources_output],
                concurrency_limit=3
            )
            
            clear_btn.click(
                fn=lambda: ([], pd.DataFrame(), {"queries_processed": 0, "avg_response_time": 0, "tokens_processed": 0}),
                outputs=[chatbot, sources_output, metrics_output]
            )
            
            export_btn.click(
                fn=self.export_session,
                outputs=export_data
            ).then(
                fn=self._save_export,
                inputs=export_data,
                outputs=export_file
            )
            
            # Update metrics and history on any response
            submit_event.then(
                fn=lambda: {
                    "queries_processed": self.performance_metrics["query_count"],
                    "avg_response_time": round(self.performance_metrics["avg_response_time"], 2),
                    "tokens_processed": self.performance_metrics["total_tokens_processed"]
                },
                outputs=metrics_output
            ).then(
                fn=self._get_history_preview,
                outputs=history_output
            )
            
            # Filter sources by confidence
            filter_btn.click(
                fn=lambda df, threshold: df[df['Confidence'].astype(float) >= threshold],
                inputs=[sources_output, confidence_slider],
                outputs=sources_output
            )

            # Launch parameters
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                max_threads=40,
                favicon_path=str(Path(__file__).parent/"assets/favicon.ico") if (Path(__file__).parent/"assets/favicon.ico").exists() else None,
                show_error=True
            )

if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        logger.info("Initializing vector store...")
        vector_store = get_vector_store()
        
        logger.info("Starting RAG system...")
        rag_system = EliteRAGSystem(vector_store)
        
        logger.info("Launching enhanced interface...")
        EliteChatInterface(rag_system).launch()
    except Exception as e:
        logger.critical(f"Failed to start: {str(e)}", exc_info=True)
        raise