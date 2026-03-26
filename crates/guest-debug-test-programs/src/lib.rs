use wstd::http::{Body, Request, Response};

#[wstd::http_server]
async fn main(_request: Request<Body>) -> anyhow::Result<Response<Body>> {
    Ok(Response::new("Hello from guest-debug test!\n".into()))
}
