use crate::models::user::{CreateUserRequest, CreateUserResponse};

/// 这里是领域逻辑的入口。示例没有数据库，因此只构造返回值。
pub fn create_user(request: CreateUserRequest) -> CreateUserResponse {
    CreateUserResponse {
        id: "demo-user-1".to_owned(),
        name: request.name,
        email: request.email,
    }
}
