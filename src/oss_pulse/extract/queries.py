"""SQL query templates for GH Archive on BigQuery."""

TOP_REPOS_QUERY = """
SELECT
    repo.name AS repo_name,
    COUNT(DISTINCT actor.login) AS contributors,
    COUNT(*) AS event_count
FROM `githubarchive.day.2025*`
WHERE type = 'PullRequestEvent'
GROUP BY repo_name
ORDER BY contributors DESC
LIMIT @top_n
"""

PR_EVENTS_QUERY = """
SELECT
    repo.name AS repo_name,
    JSON_EXTRACT_SCALAR(payload, '$.number') AS pr_number,
    JSON_EXTRACT_SCALAR(payload, '$.action') AS action,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.state') AS state,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') AS merged,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at') AS pr_created_at,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at') AS pr_merged_at,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.closed_at') AS pr_closed_at,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.updated_at') AS pr_updated_at,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') AS author,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.additions') AS additions,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.deletions') AS deletions,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.changed_files') AS changed_files,
    actor.login AS event_actor,
    created_at AS event_timestamp
FROM `githubarchive.day.*`
WHERE
    type = 'PullRequestEvent'
    AND repo.name IN UNNEST(@repo_list)
    AND _TABLE_SUFFIX BETWEEN @start_date AND @end_date
"""

PR_REVIEW_EVENTS_QUERY = """
SELECT
    repo.name AS repo_name,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS pr_number,
    actor.login AS reviewer,
    created_at AS review_timestamp
FROM `githubarchive.day.*`
WHERE
    type = 'PullRequestReviewEvent'
    AND repo.name IN UNNEST(@repo_list)
    AND _TABLE_SUFFIX BETWEEN @start_date AND @end_date
"""
