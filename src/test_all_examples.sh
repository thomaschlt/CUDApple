#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'


PASSED=()
FAILED=()
TOTAL=0

echo -e "${BLUE}╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸${NC}"
echo -e "${BLUE}   CUDApple Examples Test Suite${NC}"
echo -e "${BLUE}╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸${NC}"
echo

EXAMPLES=(
    "vector_add.cu"
    "matrix_add.cu"
    "simple_relu.cu"
    "activation_functions.cu"
    "sgd_optimizer.cu"
    "linear.cu"
    "linear_backward.cu"
    "linear_backward_bias.cu" 
    "linear_backward_weights.cu"
    "conv2d.cu"
    "conv2d_backward_weights.cu"
    "conv2d_backward_bias.cu"
    "maxpool2d.cu"
    "softmax.cu"
    "softmax_backward.cu"
    "cross_entropy_loss.cu"
    "math_functions_test.cu"
)

test_example() {
    local example="$1"
    local index="$2"
    local total="$3"
    
    echo -e "${YELLOW}[$index/$total] Testing: $example${NC}"
    echo "----------------------------------------"
    
    # Run the example
    if cargo run -- -i "examples/$example" -d output --run -v > "test_results_${example}.log" 2>&1; then
        if grep -q "Successfully completed all operations" "test_results_${example}.log"; then
            echo -e "${GREEN}✅ PASSED: $example${NC}"
            PASSED+=("$example")
        else
            echo -e "${RED}❌ FAILED: $example (completed but with issues)${NC}"
            FAILED+=("$example")
            echo "   Check test_results_${example}.log for details"
        fi
    else
        echo -e "${RED}❌ FAILED: $example (execution error)${NC}"
        FAILED+=("$example")
        echo "   Check test_results_${example}.log for details"
    fi
    
    echo
    ((TOTAL++))
}

rm -f test_results_*.log

for i in "${!EXAMPLES[@]}"; do
    test_example "${EXAMPLES[$i]}" "$((i+1))" "${#EXAMPLES[@]}"
done

echo -e "${BLUE}╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸${NC}"
echo -e "${BLUE}   Test Results Summary${NC}"
echo -e "${BLUE}╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸${NC}"
echo

echo -e "${GREEN}✅ PASSED (${#PASSED[@]}/${TOTAL}):${NC}"
for example in "${PASSED[@]}"; do
    echo "   • $example"
done

echo
echo -e "${RED}❌ FAILED (${#FAILED[@]}/${TOTAL}):${NC}"
for example in "${FAILED[@]}"; do
    echo "   • $example"
done

if [ ${#FAILED[@]} -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All tests passed! CUDApple is working perfectly!${NC}"
else
    echo -e "\n${YELLOW}📋 Check individual log files for failed tests:${NC}"
    for example in "${FAILED[@]}"; do
        echo "   • test_results_${example}.log"
    done
fi