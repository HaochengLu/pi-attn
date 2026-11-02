use axum::{Json, http::StatusCode};
use axum_valid::Valid;

use crate::{
    models::user::{CreateUserRequest, CreateUserResponse},
    services::user_service,
};

/// 创建一个示例用户。
///
/// `Valid` 会在 controller 运行前调用 `CreateUserRequest::validate`；
/// 校验失败时由 Axum 返回 JSON 格式的 400 响应，handler 不会被执行。
#[utoipa::path(
    post,
    path = "/users",
    tag = "users",
    request_body = CreateUserRequest,
    responses(
        (status = 201, description = "用户创建成功", body = CreateUserResponse),
        (status = 400, description = "JSON 格式或字段校验失败")
    )
)]
pub async fn create_user(
    Valid(Json(request)): Valid<Json<CreateUserRequest>>,
) -> (StatusCode, Json<CreateUserResponse>) {
    let user = user_service::create_user(request);
    (StatusCode::CREATED, Json(user))
}
