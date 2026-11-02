use crate::services::health_service;

/// `GET /`：返回服务欢迎信息。
pub async fn hello() -> &'static str {
    health_service::greeting()
}

/// `GET /health`：供负载均衡或监控系统检查服务状态。
pub async fn health() -> &'static str {
    health_service::status()
}
