# Friend Recommendation Algorithm

## Algorithm Overview

This algorithm uses MapReduce with Spark RDDs to recommend friends based on mutual friend counts. Two users who share many mutual friends but are not yet connected are good candidates for recommendation.

---

## Algorithm Steps

1. **Parse Input Data**

   - Read user-friend adjacency list
   - Parse each line into (UserID, [FriendList])

2. **Sample Users**

   - Randomly select 10 users for recommendations
   - Broadcast sampled user set to all workers

3. **Identify Existing Friendships**

   - For sampled users, extract all (user, friend) pairs
   - Normalize pairs to handle undirected edges: (A,B) = (B,A)

4. **Count Mutual Friends**

   - For each user's friend list, generate all friend pairs
   - Each pair shares that user as a mutual friend
   - Aggregate counts across all users using reduceByKey

5. **Filter Non-Friends**

   - Remove pairs that are already friends (leftOuterJoin)
   - Keep only candidate pairs with mutual friends

6. **Build Recommendations**

   - Convert pair-centric to user-centric format
   - For each user, sort candidates by mutual friend count (descending)
   - Select top 10 recommendations per user

7. **Output Results**
   - Display recommendations in format: UserID\trecommendations

---

## Pseudo-Code

```
FUNCTION recommend_friends(input_file, num_samples, seed):

    // Step 1: Load and parse data
    user_friends = LOAD(input_file)
                   .MAP(parse_line)
                   .FILTER(valid entries)
                   .CACHE()

    // Step 2: Sample users
    sampled_users = SAMPLE(user_friends, num_samples, seed)

    // Step 3: Get existing friendships
    existing_edges = user_friends
                     .FILTER(user in sampled_users)
                     .FLATMAP(user -> [(user, friend) for friend in friends])
                     .MAP(normalize_pair)
                     .DISTINCT()

    // Step 4: Count mutual friends
    mutual_counts = user_friends
                    .FLATMAP(generate_friend_pairs)
                    .REDUCEBYKEY(sum)

    // Step 5: Remove existing friends
    candidates = mutual_counts
                 .LEFTOUTERJOIN(existing_edges)
                 .FILTER(not already friends)

    // Step 6: Build top-10 recommendations
    recommendations = candidates
                      .FLATMAP(expand_to_users)
                      .GROUPBYKEY()
                      .MAPVALUES(sort and take top 10)

    // Step 7: Display results
    COLLECT and PRINT(recommendations)

// Helper functions
FUNCTION normalize_pair(a, b):
    RETURN (min(a,b), max(a,b))

FUNCTION generate_friend_pairs(user, friends):
    FOR each pair (f_i, f_j) in friends where i < j:
        EMIT (normalize_pair(f_i, f_j), 1)

FUNCTION expand_to_users(pair, count):
    IF pair[0] in sampled_users:
        EMIT (pair[0], (pair[1], count))
    IF pair[1] in sampled_users:
        EMIT (pair[1], (pair[0], count))
```

---

## Complexity Analysis

- **Time:** O(N × F²) where N = users, F = avg friends per user
- **Space:** O(E) where E = number of edges in friendship graph
- **MapReduce Stages:** 4 main shuffles (distinct, reduceByKey, join, groupByKey)

---

## Key MapReduce Operations

1. **Map:** Parse lines, generate pairs
2. **FlatMap:** Expand friend lists to pairs
3. **ReduceByKey:** Aggregate mutual friend counts
4. **GroupByKey:** Collect candidates per user
5. **Join:** Filter existing friendships
