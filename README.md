# Comparing LLM Prompting Frameworks: OpenAI+Instructor vs. DSPy for Business Intelligence Queries

## Overview

This project evaluates the effectiveness of different Large Language Model (LLM) prompting frameworks for analyzing natural language queries related to business intelligence (BI) data. The primary goal was to compare the performance of the currently used OpenAI Chat Completion protocol combined with the Instructor package against the DSPy framework. The evaluation provides insights for Spotonix, Inc. on potentially migrating to DSPy for improved query analysis.

## Context

Spotonix, Inc. specializes in transforming business data into accessible natural language query formats. They leverage advanced BI and data analytics to help enterprises interact with their data more effectively. This project explores ways to enhance their core technology by evaluating alternative LLM prompting frameworks.

## Frameworks Evaluated

1.  **OpenAI Chat Completion + Instructor:** The existing approach used by Spotonix, combining OpenAI's API with the Instructor library for structured output (schema enforcement).
2.  **DSPy Framework:** A framework from Stanford NLP for algorithmically optimizing LLM prompts and weights within complex pipelines, analogous to how PyTorch optimizes neural networks.

## Methodology

### Dataset

* **TPCDS Benchmark:** This dataset was used for evaluation. It includes definitions and queries in both natural language and SQL, suitable for testing BI query analysis.
* **Query Types:** The evaluation included queries targeting multiple fact tables and dimension prompts to assess performance in complex scenarios.

### Implementation Approaches

* **OpenAI + Instructor:**
    * Defined Python classes (`LLMResponseFactTables`, `LLMResponseDimensionTables`, `FactPhrase`, `DimensionPhrase`) using `OpenAISchema` to structure the LLM's output.
    * These schemas capture relevant fact/dimension table names and the corresponding phrases from the user's natural language query.
    * Utilized the `gpt-3.5-turbo` model via the OpenAI API and Instructor's `client.chat.completions.create` method with the specified `response_model`.

* **DSPy:**
    * Defined similar Python classes for structuring output.
    * Used `dspy.Signature` to define the input (user query) and expected output format (structured list of facts/dimensions and phrases).
    * Created a `dspy.Module` using `dspy.functional.TypedPredictor` to encapsulate the prompt and prediction logic.
    * Configured DSPy (`dspy.settings.configure`) to use the same underlying LLM (`gpt-3.5-turbo`).

## Key Findings & Results

* **Accuracy:** DSPy consistently produced significantly more accurate and comprehensive results compared to the OpenAI+Instructor setup, despite both using the `gpt-3.5-turbo` model.
* **Fact Table Example:** In one instance, DSPy correctly identified three relevant fact tables ('store sales data', 'store returns', 'catalog sales') and corresponding phrases, while OpenAI+Instructor missed the 'catalog sales' fact.
* **Dimension Table Example:** DSPy demonstrated superior performance in identifying necessary dimensions and accurately capturing relevant phrases from the query, whereas OpenAI+Instructor overlooked important dimensions and had issues with phrase mapping.
* **Consistency:** These observations held true across multiple queries tested using the TPCDS benchmark.

## Conclusion

The comparative analysis provides strong evidence that the DSPy framework offers a superior approach for handling complex business intelligence queries compared to the OpenAI+Instructor combination. DSPy delivers more accurate and complete results using the same base LLM and computational resources, making it a compelling alternative for Spotonix to enhance their natural language query analysis capabilities.

*(Detailed results and code are available on the project's GitHub repository as mentioned in the report.)*

## References Mentioned in Report

* Stanford NLP - DSPy
* OpenAI API
* Instructor Protocol API
* TPCDS Benchmark Dataset
