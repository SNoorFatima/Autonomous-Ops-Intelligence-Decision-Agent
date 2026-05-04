# Industrial Packaging & Deployment Strategy

## 1. Reproducible Container Image
- **Base Image Justification:** The system utilizes `python:3.11-slim` as its base image. This strikes the perfect balance between minimizing the attack surface and download size (unlike the full `python:3.11` image) while maintaining full compatibility with Python packages that might need compilation (which can be problematic in Alpine-based images).
- **Layer Ordering Strategy:** The `Dockerfile` is optimized to leverage Docker's layer caching mechanism. System dependencies (`apt-get`) are installed first, followed by copying `requirements.txt` and running `pip install`. The application source code is copied last. Because application code changes much more frequently than dependencies, this ordering prevents the time-consuming `pip install` step from running on every rebuild unless `requirements.txt` itself is modified.
- **Reproducibility:** By relying exclusively on `requirements.txt` and explicit environment variables injected at runtime, the image guarantees identical behavior across different host operating systems and local environments.

## 2. Secret-Free Image
- **Build-Time vs. Runtime:** The `Dockerfile` does not copy any `.env` files or hardcoded credentials into the image. All secrets are explicitly excluded.
- **Runtime Injection:** The `GROQ_API_KEY` is passed to the container purely at runtime via the `docker-compose.yml` environment mapping: `- GROQ_API_KEY=${GROQ_API_KEY}`. This means the image can be safely pushed to a public registry without exposing sensitive keys.
- **Exclusion of Unnecessary Files:** Local caches like `__pycache__` and `venv/` are kept out of the image by keeping them off the build path (or via `.dockerignore` conventions). The database is decoupled into its own service.

## 3. Multi-Service Orchestration
- **Service Definitions:** The system is orchestrated using Docker Compose with two distinct services:
  1. `agent-api`: The FastAPI web server executing the LangGraph agent logic.
  2. `chroma-db`: The official standalone Chroma vector database.
- **Service Discovery & Startup:** The `docker-compose.yml` uses the `depends_on` directive to ensure `chroma-db` starts before the `agent-api`. The agent discovers the database via the injected `CHROMA_HOST` environment variable, resolving to the container name `chroma-db` on the internal Docker network.
- **Data Persistence:** The vector data must survive container restarts. To achieve this, a named volume `chroma-data` is mapped to `/chroma/chroma` within the `chroma-db` container. This ensures that the RAG index and memory checkpoints are permanently stored on the host machine, immune to container lifecycle events.

## 4. End-to-End Test
*Note: Evidence of the successful deployment (build logs, curl tests) will be provided as part of the execution walkthrough upon verifying the docker-compose stack.*
