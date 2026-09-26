//! Standalone example of using HTTP embeddings configured via environment.
//! 
//! Requires the `http` feature:
//! `cargo run --features http --example http_embedding`
//! 
//! Note: Providing a real key and endpoint may incur provider charges.

use std::env;
use agent_memory::{
    http_embed::OpenAiEmbedder,
    sqlite_store::SqliteStore,
    types::Scope,
    AgentMemory,
    RetrievalConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = match env::var("OPENAI_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            println!("Usage:");
            println!("  OPENAI_API_KEY=your-key OPENAI_MODEL=text-embedding-3-small OPENAI_DIM=1536 OPENAI_ENDPOINT=https://api.openai.com/v1/embeddings cargo run --features http --example http_embedding");
            println!("\nSkipping real run because OPENAI_API_KEY is not set.");
            return Ok(());
        }
    };

    let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "text-embedding-3-small".to_string());
    let dim_str = env::var("OPENAI_DIM").unwrap_or_else(|_| "1536".to_string());
    let dim = dim_str.parse::<usize>().map_err(|_| "OPENAI_DIM must be an integer")?;
    let endpoint = env::var("OPENAI_ENDPOINT").unwrap_or_else(|_| "https://api.openai.com/v1/embeddings".to_string());

    if endpoint.starts_with("http://") && !endpoint.contains("localhost") && !endpoint.contains("127.0.0.1") {
        return Err("Use HTTPS for remote endpoints. Loopback HTTP is only for local development.".into());
    }

    println!("Initializing HTTP embedder with model: {} (dim: {}) at endpoint: {}", model, dim, endpoint);
    let embedder = OpenAiEmbedder::new(api_key, model, dim).with_endpoint(endpoint);

    // Provide a fresh, explicit demo database
    let db_path = env::temp_dir().join(format!("agent-memory-http-example-{}.db", std::process::id()));
    
    // Note: Stored and query embeddings must use a compatible model/dimension.
    let store = Box::new(SqliteStore::open(&db_path.to_string_lossy())?);
    let memory = AgentMemory::with_store(
        store,
        std::sync::Arc::new(embedder),
        RetrievalConfig::default()
    );

    println!("Database created at: {:?}", db_path);

    println!("Adding fact...");
    memory.remember_fact(Scope::User, "alice", "Alice lives in Paris and loves pastries.")?;

    println!("Querying fact...");
    let hits = memory.recall(Scope::User, "alice", "Where does she live?")?;
    if let Some(hit) = hits.first() {
        println!("Top hit: {}", hit.item.content);
    }

    // Clean up
    let _ = std::fs::remove_file(db_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_memory::http_embed::HttpTransport;
    use agent_memory::StoreResult;

    struct MockTransport;
    impl HttpTransport for MockTransport {
        fn post_json(
            &self,
            _url: &str,
            _bearer: &str,
            _body_json: &str,
        ) -> StoreResult<String> {
            Ok(r#"{
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]}
                ]
            }"#
            .to_string())
        }
    }

    #[test]
    fn test_example_offline_wiring() {
        let embedder = OpenAiEmbedder::new("fake-key", "test-model", 3)
            .with_endpoint("http://localhost/v1/embeddings")
            .with_transport(MockTransport);

        let db_path = env::temp_dir().join(format!("agent-memory-http-example-test-{}.db", std::process::id()));
        
        let store = Box::new(SqliteStore::open(&db_path.to_string_lossy()).unwrap());
        let memory = AgentMemory::with_store(
            store,
            std::sync::Arc::new(embedder),
            RetrievalConfig::default()
        );

        memory.remember_fact(Scope::User, "alice", "Alice lives in Paris").unwrap();
        let hits = memory.recall(Scope::User, "alice", "Where?").unwrap();
        
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].item.content, "Alice lives in Paris");

        let _ = std::fs::remove_file(db_path);
    }
}
