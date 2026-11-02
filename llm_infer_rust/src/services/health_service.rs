/// 提供服务的业务信息；目前不需要访问数据库或其他外部资源。
pub fn greeting() -> &'static str {
    "Hello from Axum!\n"
}

/// 返回健康检查结果。
pub fn status() -> &'static str {
    "ok\n"
}
