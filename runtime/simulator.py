from typing import Dict, Any, List

class RuntimeSimulator:
    def __init__(self):
        pass

    def simulate_execution(self, schemas: Dict[str, Any]) -> Dict[str, Any]:
        results = {
            "can_execute": True,
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
            "execution_plan": [],
            "estimated_performance": {}
        }

        # Database checks
        db_check = self._check_database(schemas.get('db', {}))
        if db_check['valid']:
            results['checks_passed'].append("Database schema is valid")
        else:
            results['checks_failed'].extend(db_check['errors'])
            results['can_execute'] = False
        results['warnings'].extend(db_check.get('warnings', []))

        # API checks
        api_check = self._check_api(schemas.get('api', {}))
        if api_check['valid']:
            results['checks_passed'].append("API routes are valid")
        else:
            results['checks_failed'].extend(api_check['errors'])
            results['can_execute'] = False

        # UI checks
        ui_check = self._check_ui(schemas.get('ui', {}))
        if ui_check['valid']:
            results['checks_passed'].append("UI components are valid")
        else:
            results['warnings'].extend(ui_check.get('errors', []))

        # Auth checks
        auth_check = self._check_auth(schemas.get('auth', {}))
        if auth_check['valid']:
            results['checks_passed'].append("Auth rules are consistent")
        else:
            results['checks_failed'].extend(auth_check['errors'])
            results['can_execute'] = False

        results['execution_plan'] = self._generate_execution_plan(schemas)
        results['estimated_performance'] = self._estimate_performance(schemas)

        return results

    def _check_database(self, db_schema: Any) -> Dict:
        errors = []
        warnings = []
        if not isinstance(db_schema, dict):
            errors.append(f"Database schema is not a dict (type: {type(db_schema).__name__})")
            return {"valid": False, "errors": errors, "warnings": warnings}

        tables = db_schema.get('tables')
        if not isinstance(tables, dict):
            errors.append("Database 'tables' is missing or not a dict")
            return {"valid": False, "errors": errors, "warnings": warnings}

        if not tables:
            warnings.append("No database tables defined")

        for table_name, table_config in tables.items():
            if not isinstance(table_config, dict):
                errors.append(f"Table '{table_name}' config is not a dict (type: {type(table_config).__name__})")
                continue

            fields = table_config.get('fields')
            if not isinstance(fields, dict):
                errors.append(f"Table '{table_name}' has no 'fields' dict")
                continue

            # Check each field
            has_primary = False
            for field_name, field_props in fields.items():
                if not isinstance(field_props, dict):
                    errors.append(f"Field '{field_name}' in '{table_name}' is not a dict")
                    continue
                if field_props.get('primary_key'):
                    has_primary = True
            if not has_primary:
                warnings.append(f"Table '{table_name}' has no primary key column")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _check_api(self, api_schema: Any) -> Dict:
        errors = []
        if not isinstance(api_schema, dict):
            errors.append(f"API schema is not a dict (type: {type(api_schema).__name__})")
            return {"valid": False, "errors": errors}

        endpoints = api_schema.get('endpoints')
        if not isinstance(endpoints, list):
            errors.append("API 'endpoints' is missing or not a list")
            return {"valid": False, "errors": errors}

        if not endpoints:
            errors.append("No API endpoints defined")

        paths_seen = set()
        valid_methods = {'GET', 'POST', 'PUT', 'DELETE', 'PATCH'}
        for idx, endpoint in enumerate(endpoints):
            if not isinstance(endpoint, dict):
                errors.append(f"Endpoint at index {idx} is not a dict")
                continue
            path = endpoint.get('path')
            method = endpoint.get('method', '').upper()
            if not path:
                errors.append(f"Endpoint {idx} missing 'path'")
            if not method:
                errors.append(f"Endpoint {idx} missing 'method'")
            elif method not in valid_methods:
                errors.append(f"Endpoint {idx} has invalid method '{method}'")
            key = (path, method)
            if key in paths_seen:
                errors.append(f"Duplicate endpoint: {method} {path}")
            paths_seen.add(key)

        return {"valid": len(errors) == 0, "errors": errors}

    def _check_ui(self, ui_schema: Any) -> Dict:
        errors = []
        if not isinstance(ui_schema, dict):
            errors.append(f"UI schema is not a dict (type: {type(ui_schema).__name__})")
            return {"valid": False, "errors": errors}

        pages = ui_schema.get('pages')
        if not isinstance(pages, dict):
            errors.append("UI 'pages' is missing or not a dict")
        elif not pages:
            errors.append("No UI pages defined")

        routing = ui_schema.get('routing')
        if not isinstance(routing, dict):
            errors.append("UI 'routing' is missing or not a dict")
        elif not routing:
            errors.append("No routing configuration")

        return {"valid": len(errors) == 0, "errors": errors}

    def _check_auth(self, auth_schema: Any) -> Dict:
        errors = []
        if not isinstance(auth_schema, dict):
            errors.append(f"Auth schema is not a dict (type: {type(auth_schema).__name__})")
            return {"valid": False, "errors": errors}

        roles = auth_schema.get('roles')
        if not isinstance(roles, dict):
            errors.append("Auth 'roles' is missing or not a dict")
        elif not roles:
            errors.append("No roles defined")

        return {"valid": len(errors) == 0, "errors": errors}

    def _generate_execution_plan(self, schemas: Dict) -> List[Dict]:
        db_tables = len(schemas.get('db', {}).get('tables', {}))
        api_endpoints = len(schemas.get('api', {}).get('endpoints', []))
        ui_pages = len(schemas.get('ui', {}).get('pages', {}))
        return [
            {"step": 1, "action": "Initialize database", "details": f"Create {db_tables} tables and relationships", "status": "pending"},
            {"step": 2, "action": "Setup API server", "details": f"Create {api_endpoints} endpoints and middleware", "status": "pending"},
            {"step": 3, "action": "Configure auth system", "details": "Setup roles and permissions", "status": "pending"},
            {"step": 4, "action": "Generate frontend", "details": f"Create {ui_pages} pages and components", "status": "pending"}
        ]

    def _estimate_performance(self, schemas: Dict) -> Dict:
        db_tables = len(schemas.get('db', {}).get('tables', {}))
        api_endpoints = len(schemas.get('api', {}).get('endpoints', []))
        ui_pages = len(schemas.get('ui', {}).get('pages', {}))
        complexity_score = db_tables + api_endpoints + ui_pages

        if complexity_score < 10:
            performance = "Fast response times (<100ms)"
        elif complexity_score < 30:
            performance = "Good performance (100-300ms)"
        else:
            performance = "May need optimization (300ms+)"

        return {
            "complexity_score": complexity_score,
            "estimated_performance": performance,
            "db_tables": db_tables,
            "api_endpoints": api_endpoints,
            "ui_pages": ui_pages
        }
