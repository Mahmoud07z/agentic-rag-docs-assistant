# Atlas Notes API

This small sample document is included to demonstrate local ingestion. Replace it with documentation you want the assistant to answer questions about.

## Authentication

Send an API key in the `Authorization` header using the `Bearer` scheme. Requests without a valid key receive HTTP 401. Keep API keys private and do not include them in source control.

## Creating a note

Create a note with `POST /v1/notes` and a JSON body containing a non-empty `title` and `body`. A successful request returns HTTP 201 and includes the new note's identifier and creation time.

## Rate limits

The API allows 60 requests per minute per API key. When the limit is exceeded, the server returns HTTP 429 and a `Retry-After` header indicating how many seconds to wait before retrying.
