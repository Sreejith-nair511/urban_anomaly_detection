from pydantic import BaseModel, Field

class PredictInput(BaseModel):
    AQI:             float = Field(..., ge=0,  le=500,  example=120)
    temperature:     float = Field(..., ge=0,  le=60,   example=28)
    humidity:        float = Field(..., ge=0,  le=100,  example=65)
    traffic_density: float = Field(..., ge=0,  le=100,  example=45)
    noise_level:     float = Field(..., ge=0,  le=150,  example=70)
    hour:            int   = Field(default=12, ge=0, le=23)
    day_of_week:     int   = Field(default=0,  ge=0, le=6)
