import json
from typing import Type, Optional, List, Dict, Any
from pydantic import BaseModel, create_model, Field
from loguru import logger
import litellm

class JudgeFeedback(BaseModel):
    requirements_met: bool = Field(description="True if all global requirements, criteria guidelines, and individual variations match.")
    critique_report: str = Field(description="Detailed diagnostic overview explaining specific adherence issues or gaps found.")

class SyntheticDataPipeline:
    def __init__(self):
        litellm.enable_json_schema_validation = True

    def _build_pydantic_field(self, field_info: Dict[str, Any]) -> Any:
        type_str = field_info.get("type", "string").lower()
        desc = field_info.get("description", "")

        if type_str == "object" and "properties" in field_info:
            nested_fields = {
                sub_name: self._build_pydantic_field(sub_info)
                for sub_name, sub_info in field_info["properties"].items()
            }
            nested_model = create_model("NestedBlock", **nested_fields)
            return (nested_model, Field(default=..., description=desc))

        type_mapping = {"integer": int, "number": float, "boolean": bool, "string": str}
        return (type_mapping.get(type_str, str), Field(default=..., description=desc))

    def compile_schema_to_pydantic(self, schema_dict: Dict[str, Any]) -> Type[BaseModel]:
        fields = {
            field_name: self._build_pydantic_field(field_info)
            for field_name, field_info in schema_dict.items()
        }
        return create_model("DynamicRecord", **fields)

    def run_generation(self, gen_model: str, user_requirements: str, schema_dict: Dict[str, Any], volume: int, output_file: str) -> List[BaseModel]:
        """Generates dynamic records strictly aligning to global rules and field definitions."""
        logger.info(f"Launching dataset generation via {gen_model} for {volume} records.")
        
        DynamicRecord = self.compile_schema_to_pydantic(schema_dict)
        class DataContainer(BaseModel):
            dataset: List[DynamicRecord]

        # The core change: Explicitly forcing the LLM to blend structural schema and qualitative parameters
        prompt = f"""
        You are a production-grade synthetic data generator. 
        Your primary task is to generate exactly {volume} highly varied, unique records.
        
        ### CRITICAL OPERATION RULES (USER INSTRUCTIONS):
        {user_requirements}
        
        ### TARGET SCHEMA STRUCTURE & FIELD DESCRIPTIONS:
        {json.dumps(schema_dict, indent=2)}
        
        Generate rows that perfectly execute the guidelines above while maintaining strict datatype alignment.
        """

        response = litellm.completion(
            model=gen_model,
            messages=[{"role": "user", "content": prompt}],
            response_format=DataContainer
        )
        
        parsed_container = DataContainer.model_validate_json(response.choices[0].message.content)
        
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in parsed_container.dataset:
                f.write(entry.model_dump_json() + "\n")
                
        logger.success(f"Successfully generated and stored {len(parsed_container.dataset)} records inside {output_file}")
        return parsed_container.dataset

    def run_post_audit(self, judge_model: str, user_requirements: str, schema_dict: Dict[str, Any], generated_records: List[BaseModel]) -> Dict[str, Any]:
        logger.info(f"Dispatching post-generation audit evaluation to Judge: {judge_model}")
        
        serialized_data = "\n".join([rec.model_dump_json() for rec in generated_records])
        
        prompt = f"""
        You are an independent data auditor. Evaluate the generated synthetic data records.
        
        Expected Goals: {user_requirements}
        Defined Layout Metrics: {json.dumps(schema_dict, indent=2)}
        
        Generated Data Sample:
        {serialized_data}
        
        Analyze if the generated fields accurately adhere to individual row instructions and if semantic variety rules are broken.
        """
        
        response = litellm.completion(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format=JudgeFeedback
        )
        return JudgeFeedback.model_validate_json(response.choices[0].message.content).model_dump()