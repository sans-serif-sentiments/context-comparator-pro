---
id: extraction_json
difficulty: hard
goal: structured data extraction
metrics: [faithfulness, completeness, structure]
expected_structure: json
-----------------------------------------------

Prompt:
Extract the following fields as JSON from the customer feedback paragraph below: `customer_name`, `issue_category`, `severity` (low|medium|high), and `recommended_action`. Ensure the output is valid JSON and contains only those keys.

Paragraph:
Shayla Ortiz reported that invoices generated on 2024-02-17 displayed duplicate line items after the latest deployment. Finance cannot close the books unless a rollback or hotfix is shipped today.

