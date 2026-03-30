# Task Title

## Module

Select the module impacted by this task:

* data
* preprocessing
* datasets
* models
* training
* evaluation
* inference
* api
* app
* experiments
* docs
* configs

---

## Task Description

Provide a clear functional description of the task.

Example:

Prepare paired datasets from original face images by generating masked and low-resolution versions suitable for supervised reconstruction training.

---

## Objective

Explain what the task should achieve.

Example:

Create a reproducible preprocessing pipeline that generates input-output image pairs for model training.

---

## Expected Inputs

List required inputs if applicable.

Example:

* Raw face dataset
* Image size configuration
* Mask generation parameters

---

## Expected Outputs

Describe expected deliverables.

Example:

* Masked images
* Low-resolution images
* Paired dataset directory
* Dataset split (train/validation/test)

---

## Functional Requirements

Describe expected system behavior.

Example:

* Images must be resized consistently
* Masks must be applied automatically
* Dataset pairing must remain reproducible
* Output structure must follow repository conventions

---

## Constraints

List constraints if any.

Example:

* Must support configurable image sizes
* Must not modify raw dataset files
* Must remain compatible with training module

---

## Dependencies

List modules or tasks this depends on.

Example:

Requires dataset download pipeline to be completed first.

---

## Assigned Engineer

Select responsible role:

* Data Pipeline Engineer
* Model Training Engineer
* Evaluation & Deployment Engineer

---

## Priority Level

Select one:

* High
* Medium
* Low

---

## Definition of Done

Task is considered complete when:

* functionality implemented
* module tested independently
* outputs verified
* compatible with project architecture
* ready for merge into main branch

---

## Suggested Branch Name

Example:

feature/preprocessing-mask-generator

---

## Notes (Optional)

Add references, dataset links, research ideas, or experimentation suggestions.
