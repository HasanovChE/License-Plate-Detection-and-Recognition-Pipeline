def normalize_text(text):
    return (
        text.upper()
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
    )

def exact_match(predicted, ground_truth):
    predicted = normalize_text(predicted)
    ground_truth = normalize_text(ground_truth)
    return predicted == ground_truth

def character_error_rate(predicted, truth): 
    predicted = normalize_text(predicted) 
    truth = normalize_text(truth) 
 
    m = len(truth) 
    n = len(predicted) 
 
    dp = [ 
        [0] * (n + 1) 
        for _ in range(m + 1) 
    ] 
 
    for i in range(m + 1): 
        dp[i][0] = i 
 
    for j in range(n + 1): 
        dp[0][j] = j 
 
    for i in range(1, m + 1): 
        for j in range(1, n + 1): 
            cost = ( 
                0 
                if truth[i - 1] == predicted[j - 1] 
                else 1 
            ) 
 
            dp[i][j] = min( 
                dp[i - 1][j] + 1, 
                dp[i][j - 1] + 1, 
                dp[i - 1][j - 1] + cost 
            ) 
 
    if m == 0: 
        return 0.0 if n == 0 else 1.0 
 
    return dp[m][n] / m