from pydantic import BaseModel, Field


class AdmissionRequest(BaseModel):

    GRE_Score: int = Field(..., ge=260, le=340)
    TOEFL_Score: int = Field(..., ge=0, le=120)
    University_Rating: int = Field(..., ge=1, le=5)
    SOP: float = Field(..., ge=1, le=5)
    LOR: float = Field(..., ge=1, le=5)
    CGPA: float = Field(..., ge=0, le=10)
    Research: int = Field(..., ge=0, le=1)


class AdmissionResponse(BaseModel):

    chance_of_admit: float