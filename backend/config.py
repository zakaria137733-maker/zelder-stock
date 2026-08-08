from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongo_uri: str = "mongodb://localhost:27017/sentimentiq"
    influx_url: str = "http://localhost:8086"
    influx_token: str = ""
    influx_org: str = "sentimentiq"
    influx_bucket: str = "sentiment_scores"
    redis_url: str = "redis://localhost:6379"
    temporal_url: str = "localhost:7233"
    news_api_key: str = ""
    jwt_secret: str = ""
    admin_secret: str = ""
    admin_username: str = ""
    admin_password: str = ""
    auth_rate_limit: int = 20
    auth_rate_window_seconds: int = 300

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

settings = Settings()
