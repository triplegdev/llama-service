# Llama

## Overview
This is a LLaMA microservice that provides predictions for text inputs. It uses the pre-trained LLaMA-small model and is deployed using Docker on Render.

## Endpoints

### `/predict`
Returns predictions for a single text input.

### `/batch_predict`
Returns predictions for a batch of text inputs.

### `/healthcheck`
Returns a simple "OK" response to indicate the service is up.
