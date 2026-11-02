use axum::{
    Router,
    routing::{get, post},
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

mod health_controller;
pub mod user_controller;

#[derive(OpenApi)]
#[openapi(
    paths(user_controller::create_user),
    components(schemas(
        crate::models::user::CreateUserRequest,
        crate::models::user::CreateUserResponse
    )),
    tags((name = "users", description = "用户接口"))
)]
struct ApiDoc;

/// 注册应用的 HTTP 路由。
pub fn router() -> Router {
    Router::new()
        .route("/", get(health_controller::hello))
        .route("/health", get(health_controller::health))
        .route("/users", post(user_controller::create_user))
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
}
