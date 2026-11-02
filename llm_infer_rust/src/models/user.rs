use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

/// `POST /users` 的请求体。
#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct CreateUserRequest {
    #[validate(length(min = 3, max = 30))]
    #[schema(min_length = 3, max_length = 30, example = "alice")]
    pub name: String,

    #[validate(email)]
    #[schema(format = Email, example = "alice@example.com")]
    pub email: String,
}

/// 创建用户后的响应体。
#[derive(Debug, Serialize, ToSchema)]
pub struct CreateUserResponse {
    pub id: String,
    pub name: String,
    pub email: String,
}
