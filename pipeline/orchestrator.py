import time, logging
from typing import Dict, Any, Tuple, List
from datetime import datetime

from pipeline.intent_extractor import IntentExtractor
from pipeline.system_designer import SystemDesigner
from pipeline.schema_generator import SchemaGenerator
from pipeline.validator import Validator
from pipeline.metrics import MetricsTracker
from runtime.simulator import RuntimeSimulator

logger = logging.getLogger("ai-compiler")

class Pipeline:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.intent_extractor = IntentExtractor()
        self.system_designer = SystemDesigner()
        self.schema_generator = SchemaGenerator()
        self.validator = Validator()
        self.simulator = RuntimeSimulator()
        self.metrics = MetricsTracker()
        self.results = {}
        self.request_counter = 0

    def _safe_stage(self, stage_name: str, func_llm, func_rule, input_data, errors: List[str],
                    request_id: str) -> Any:
        """
        Execute a pipeline stage with LLM or rule‑based fallback.
        Returns the result, and appends any error to `errors` list.
        """
        try:
            logger.info(f"[{request_id}] Stage {stage_name}")
            result = func_llm(input_data) if self.use_llm else func_rule(input_data)
            self.metrics.record_stage(request_id, stage_name)
            return result
        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            errors.append(f"{stage_name}: {str(e)}")
            return func_rule(input_data)

    def compile(self, prompt: str) -> Dict[str, Any]:
        self.request_counter += 1
        request_id = f"req_{self.request_counter}_{int(datetime.now().timestamp()*1000)}"
        start_time = time.time()
        errors: List[str] = []
        warnings: List[str] = []

        self.metrics.start_timer(request_id)

        # Stage 1: Intent Extraction
        intent = self._safe_stage(
            "Intent Extraction",
            self.intent_extractor.extract,
            self.intent_extractor.extract_rule_based,
            prompt, errors, request_id
        )
        self.results["intent"] = intent

        # Stage 2: System Design
        design = self._safe_stage(
            "System Design",
            self.system_designer.design,
            self.system_designer.design_rule_based,
            intent, errors, request_id
        )
        self.results["design"] = design

        # Stage 3: Schema Generation
        schemas = self._safe_stage(
            "Schema Generation",
            self.schema_generator.generate,
            self.schema_generator.generate_rule_based,
            design, errors, request_id
        )
        self.results["schemas"] = schemas

        # Stage 4: Validation (may return (errors, warnings) or just errors)
        logger.info(f"[{request_id}] Stage 4: Validation")
        try:
            val_result = self.validator.validate_cross_layer(design, schemas)
            if isinstance(val_result, tuple) and len(val_result) == 2:
                val_errors, val_warnings = val_result
                warnings.extend(val_warnings)
            elif isinstance(val_result, list):
                val_errors = val_result
            else:
                val_errors = []
            self.metrics.record_stage(request_id, "validation_repair")
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            errors.append(f"Validation: {str(e)}")
            val_errors = []
            val_warnings = []

        # Repair count tracking (if validator has repair_count attribute)
        repair_count = getattr(self.validator, 'repair_count', 0)
        self.metrics.record_repair_attempts(request_id, repair_count)

        # Stage 5: Simulation (with its own error handling)
        logger.info(f"[{request_id}] Stage 5: Simulation")
        try:
            simulation = self.simulator.simulate_execution(schemas)
            self.metrics.record_stage(request_id, "simulation")
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            errors.append(f"Simulation: {str(e)}")
            simulation = {
                "can_execute": False,
                "checks_passed": [],
                "checks_failed": [f"Simulation error: {str(e)}"],
                "warnings": [],
                "execution_plan": [],
                "estimated_performance": {}
            }

        total_time_ms = self.metrics.end_timer(request_id)
        success = simulation.get('can_execute', False) and len(val_errors) == 0
        self.metrics.log_completion(request_id, success)

        # Combine all warnings (from validation and simulation)
        all_warnings = warnings + simulation.get('warnings', [])
        # Also include any assumptions from intent
        if intent.get("assumptions"):
            all_warnings.extend([f"Assumption: {a}" for a in intent.get("assumptions", [])])

        return {
            "request_id": request_id,
            "intent": intent,
            "design": design,
            "schemas": schemas,
            "validation": {
                "valid": len(val_errors) == 0,
                "errors": val_errors,
                "warnings": all_warnings
            },
            "simulation_result": simulation,
            "metrics": {
                "total_time_ms": total_time_ms,
                "stages": self.metrics.stage_times.get(request_id, {}),
                "repairs": repair_count
            },
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "success": success,
            "stage_errors": errors
        }

    def get_results(self) -> Dict[str, Any]:
        return self.results
