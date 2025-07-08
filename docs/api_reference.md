# API Reference

## Base URL
`https://api.creditrust.ai/v1`

## Authentication
- API Key required in `X-API-Key` header

## Endpoints

### POST /analyze
Analyze complaints based on natural language query

**Request Body:**
```json
{
  "text": "string",
  "products": ["string"],
  "date_range": ["string"],
  "advanced_options": {}
}