# import modal
# import os

# # Create Modal app
# app = modal.App("krishi-mitra")

# # Define the image with all required dependencies
# image = (
#     modal.Image.debian_slim(python_version="3.11")
#     .pip_install([
#     "gradio>=4.0.0",
#     "groq",
#     "requests",
#     "chromadb>=0.4.0",
#     "langchain>=0.1.0"
# ]).apt_install(["sqlite3"])  # System SQLite
# )

# # Create persistent volumes for data storage
# database_volume = modal.Volume.from_name("krishi-database", create_if_missing=True)
# chroma_volume = modal.Volume.from_name("krishi-chroma", create_if_missing=True)

# @app.function(
#     image=image,
#     secrets=[modal.Secret.from_name("groq-api-key")],  # You'll need to create this
#     volumes={
#         "/app/data": database_volume,
#         "/app/chroma_db": chroma_volume
#     },
#     container_idle_timeout=300,
#     timeout=3600,
#     cpu=2.0,
#     memory=4096
# )
# def run_krishi_mitra():
#     # Import your main application code here
#     import sys
#     sys.path.append("/app")
    
#     # Your existing Krishi Mitra code will be imported
#     from krishi_mitra import create_interface
    
#     # Create and launch the interface
#     app = create_interface()
    
#     # Launch with Modal-compatible settings
#     app.launch(
#         server_name="0.0.0.0",
#         server_port=8000,
#         share=False,  # Not needed on Modal
#         debug=False   # Disable debug in production
#     )

# @app.local_entrypoint()
# def main():
#     """Deploy the Krishi Mitra application"""
#     print("🚀 Deploying Krishi Mitra to Modal...")
#     run_krishi_mitra.remote()

# # For web serving
# @app.function(
#     image=image,
#     secrets=[modal.Secret.from_name("groq-api-key")],
#     volumes={
#         "/app/data": database_volume,
#         "/app/chroma_db": chroma_volume
#     },
#     container_idle_timeout=300,
#     allow_concurrent_inputs=10,
#     cpu=2.0,
#     memory=4096
# )
# @modal.web_endpoint(method="GET")
# def web():
#     """Web endpoint for the application"""
#     import sys
#     sys.path.append("/app")
    
#     from krishi_mitra import create_interface
    
#     app = create_interface()
#     return app

# # Alternative ASGI app approach (recommended for Gradio apps)
# @app.function(
#     image=image,
#     secrets=[modal.Secret.from_name("groq-api-key")],
#     volumes={
#         "/app/data": database_volume,
#         "/app/chroma_db": chroma_volume
#     },
#     container_idle_timeout=600,
#     cpu=2.0,
#     memory=4096
# )
# @modal.asgi_app()
# def fastapi_app():
#     """ASGI app for Gradio"""
#     import sys
#     sys.path.append("/app")
    
#     from krishi_mitra import create_interface
    
#     # Create Gradio interface
#     gradio_app = create_interface()
    
#     # Mount Gradio app to FastAPI
#     from fastapi import FastAPI
    
#     app = FastAPI()
    
#     # Mount Gradio app
#     app = gradio_app.mount_gradio_app(app, path="/")
    
#     return app

from krishi_mitra import create_interface