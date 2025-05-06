# SHL Assessment Recommendation System

To streamline the assessment selection process for hiring managers, developed an intelligent recommendation system that returns the most relevant assessments based on the job description query. 


## **Installing Dependencies**

The modules are added in requirements.txt. To install run,

	pip install -r requirements.txt

## **API Keys**

The following API key is required:

- [Google Gemini API Key](https://makersuite.google.com/app/apikey)

Add the key in the .env file located in project folder.

	GOOGLE_API_KEY = ADD YOUR GOOGLE GEMINI API KEY  

## **Models**

- [Text Embedding (text-embedding-004)](https://ai.google.dev/gemini-api/docs/models/gemini#text-embedding)

## Web Application

- **Frontend**  
https://mohitghai-assessmentrecommendation.streamlit.app/

- **Backend API**  
https://assessment-recommendation-wt24.onrender.com/

    - Health Check Endpoint  
    https://assessment-recommendation-wt24.onrender.com/health

    - Assessment Recommendation Endpoint  
    https://assessment-recommendation-wt24.onrender.com/recommend

    - Results  
    https://assessment-recommendation-wt24.onrender.com/metric


## API Structure & Endpoints

**Base-URL**  
https://assessment-recommendation-wt24.onrender.com

**Health Check Endpoint**  
This endpoint provides a simple status check to verify the API is running.

**Request**  

Method: GET  
Path: Base-URL/health

**Response**

    {
        "status":"healthy"
    }

**Assessment Recommendation Endpoint**  
This endpoint accepts a job description or Natural language query and returns recommended relevant assessments.

**Request**  

Method: POST  
Path: Base-URL/recommend  
Content-Type: application/json  
Body: 

    {
        "query": "Job Description"
    }

**Response**

Content-Type: application/json  
Status Code: 200 OK (if successful)  
Body:

    {
        "recommended_assessments":[
            {
                "url":"https://www.shl.com/products/product-catalog/view/java-8-new/",
                "adaptive_support":"no",
                "description":"Multi-choice test that measures the knowledge of Java class design, exceptions, generics, collections, concurrency, JDBC and Java I/O fundamentals.",
                "duration":"18",
                "remote_support":"yes",
                "test_type":["Knowledge & Skills"],
                
            },
            --   
        ]
    }


### References

- https://requests.readthedocs.io/en/latest/
- https://www.shl.com/products/product-catalog/
- https://ai.google.dev/gemini-api/docs/models/gemini#text-embedding
- https://streamlit.io/cloud
- https://render.com/
- https://docs.streamlit.io/
- https://beautiful-soup-4.readthedocs.io/en/latest/

