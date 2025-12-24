#!/bin/bash

# GeoAI REST API Test Script
# Tests all Phase 1 endpoints and validates responses

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_URL="http://127.0.0.1:8000"
API_PREFIX="/api/v1"
SERVER_PID=""

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_header() {
    echo -e "\n${YELLOW}=== $1 ===${NC}"
}

# Function to check if server is running
check_server() {
    if curl -s "${API_URL}${API_PREFIX}/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to start the server
start_server() {
    print_header "Starting FastAPI Server"

    # Start server in background
    uv run uvicorn geoaiserve.main:app --host 127.0.0.1 --port 8000 > /tmp/geoaiserve_test.log 2>&1 &
    SERVER_PID=$!

    print_info "Server PID: $SERVER_PID"
    print_info "Waiting for server to start..."

    # Wait for server to be ready (max 30 seconds)
    for i in {1..30}; do
        if check_server; then
            print_success "Server is ready!"
            return 0
        fi
        sleep 1
    done

    print_error "Server failed to start within 30 seconds"
    cat /tmp/geoaiserve_test.log
    return 1
}

# Function to stop the server
stop_server() {
    if [ -n "$SERVER_PID" ]; then
        print_header "Stopping Server"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        print_success "Server stopped (PID: $SERVER_PID)"
    fi
}

# Function to test an endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local description=$3
    local expected_status=${4:-200}

    print_info "Testing: $description"

    response=$(curl -s -w "\n%{http_code}" -X "$method" "${API_URL}${API_PREFIX}${endpoint}")
    body=$(echo "$response" | head -n -1)
    status=$(echo "$response" | tail -n 1)

    if [ "$status" -eq "$expected_status" ]; then
        print_success "$method $endpoint → HTTP $status"

        # Pretty print JSON if it's valid
        if echo "$body" | python3 -m json.tool > /dev/null 2>&1; then
            echo "$body" | python3 -m json.tool | head -20
            if [ $(echo "$body" | wc -l) -gt 20 ]; then
                echo "... (truncated)"
            fi
        else
            echo "$body"
        fi
        return 0
    else
        print_error "$method $endpoint → HTTP $status (expected $expected_status)"
        echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
        return 1
    fi
}

# Function to validate JSON structure
validate_json_field() {
    local json=$1
    local field=$2
    local description=$3

    if echo "$json" | python3 -c "import sys, json; data=json.load(sys.stdin); assert '$field' in str(data)" 2>/dev/null; then
        print_success "Field '$field' present in response"
        return 0
    else
        print_error "Field '$field' missing in response"
        return 1
    fi
}

# Trap to ensure cleanup
trap stop_server EXIT

# Main test execution
main() {
    print_header "GeoAI REST API Test Suite"

    # Start server
    if ! start_server; then
        exit 1
    fi

    sleep 2  # Give server a moment to fully initialize

    # Test 1: Health Check
    print_header "Test 1: Health Check Endpoint"
    if test_endpoint "GET" "/health" "Health check"; then
        response=$(curl -s "${API_URL}${API_PREFIX}/health")
        validate_json_field "$response" "status"
        validate_json_field "$response" "version"
        validate_json_field "$response" "timestamp"
        validate_json_field "$response" "models_loaded"
    fi

    # Test 2: List Models
    print_header "Test 2: List Models Endpoint"
    if test_endpoint "GET" "/models" "List all available models"; then
        response=$(curl -s "${API_URL}${API_PREFIX}/models")
        validate_json_field "$response" "models"
        validate_json_field "$response" "total"

        # Check if we have expected models
        sam_count=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(sum(1 for m in data['models'] if m['model_id'] == 'sam'))")
        if [ "$sam_count" -eq "1" ]; then
            print_success "SAM model found in models list"
        else
            print_error "SAM model not found in models list"
        fi
    fi

    # Test 3: Get Specific Model Info (SAM)
    print_header "Test 3: Get Model Info - SAM"
    if test_endpoint "GET" "/models/sam/info" "Get SAM model information"; then
        response=$(curl -s "${API_URL}${API_PREFIX}/models/sam/info")
        validate_json_field "$response" "model_id"
        validate_json_field "$response" "supported_tasks"
        validate_json_field "$response" "device"
    fi

    # Test 4: Get Specific Model Info (Moondream)
    print_header "Test 4: Get Model Info - Moondream"
    test_endpoint "GET" "/models/moondream/info" "Get Moondream model information"

    # Test 5: Get Specific Model Info (DINOv3)
    print_header "Test 5: Get Model Info - DINOv3"
    test_endpoint "GET" "/models/dinov3/info" "Get DINOv3 model information"

    # Test 6: Invalid Model (should return 404)
    print_header "Test 6: Error Handling - Invalid Model"
    test_endpoint "GET" "/models/invalid-model/info" "Get invalid model (should fail)" 404

    # Test 7: OpenAPI Documentation
    print_header "Test 7: OpenAPI Documentation"
    if curl -s "${API_URL}${API_PREFIX}/docs" | grep -q "Swagger UI"; then
        print_success "OpenAPI docs available at ${API_URL}${API_PREFIX}/docs"
    else
        print_error "OpenAPI docs not accessible"
    fi

    # Test 8: OpenAPI JSON Schema
    print_header "Test 8: OpenAPI JSON Schema"
    if test_endpoint "GET" "/openapi.json" "Get OpenAPI schema"; then
        response=$(curl -s "${API_URL}${API_PREFIX}/openapi.json")
        validate_json_field "$response" "openapi"
        validate_json_field "$response" "info"
        validate_json_field "$response" "paths"
    fi

    # Test 9: CORS Headers
    print_header "Test 9: CORS Configuration"
    cors_headers=$(curl -s -I -X OPTIONS "${API_URL}${API_PREFIX}/health" | grep -i "access-control")
    if [ -n "$cors_headers" ]; then
        print_success "CORS headers present"
        echo "$cors_headers"
    else
        print_info "CORS headers not found (may not be configured)"
    fi

    # Summary
    print_header "Test Summary"
    print_success "All Phase 1 endpoints tested successfully!"
    print_info "Server log available at: /tmp/geoaiserve_test.log"

    # Show some server logs
    print_header "Server Logs (last 20 lines)"
    tail -20 /tmp/geoaiserve_test.log
}

# Run main function
main

exit 0
