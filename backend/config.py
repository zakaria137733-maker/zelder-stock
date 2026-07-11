from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongo_uri: str = "mongodb://localhost:27017/sentimentiq"
    influx_url: str = "http://localhost:8086"
    influx_token: str = "sentimentiq-super-secret-token"
    influx_org: str = "sentimentiq"
    influx_bucket: str = "sentiment_scores"
    redis_url: str = "redis://localhost:6379"
    news_api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
