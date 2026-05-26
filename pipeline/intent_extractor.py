import json
import re
import logging
from typing import Dict, List, Any
from pipeline.llm import _get_nvidia_client, MODELS

logger = logging.getLogger("ai-compiler")

ENTITY_KEYWORDS = {
    'contact': 'contacts', 'user': 'users', 'customer': 'customers',
    'product': 'products', 'order': 'orders', 'invoice': 'invoices',
    'payment': 'payments', 'task': 'tasks', 'project': 'projects',
    'company': 'companies', 'lead': 'leads', 'deal': 'deals',
    'ticket': 'tickets', 'article': 'articles', 'post': 'posts',
    'comment': 'comments', 'category': 'categories', 'tag': 'tags',
    'inventory': 'inventory', 'vendor': 'vendors', 'subscription': 'subscriptions',
    'driver': 'drivers', 'vehicle': 'vehicles', 'trip': 'trips',
    'delivery': 'deliveries', 'restaurant': 'restaurants',
    'passenger': 'passengers', 'ride': 'rides',
    'rating': 'ratings', 'review': 'reviews',
    'bag': 'bags',
    'doctor': 'doctors', 'patient': 'patients', 'appointment': 'appointments',
    'clinic': 'clinics', 'prescription': 'prescriptions', 'medical_record': 'medical_records',
    'staff': 'staff', 'schedule': 'schedules',
    'map': 'maps', 'location': 'locations',
    'token': 'tokens', 'session': 'sessions',
    'notification': 'notifications', 'message': 'messages',
    'audit_log': 'audit_logs', 'audit': 'audit_logs',
}

ROLE_KEYWORDS = {
    'admin': 'admin', 'administrator': 'admin',
    'global_admin': 'global_admin', 'super_admin': 'admin',
    'manager': 'manager', 'clinic_manager': 'clinic_manager',
    'doctor': 'doctor', 'nurse': 'nurse',
    'patient': 'patient', 'customer': 'customer',
    'driver': 'driver', 'rider': 'driver',
    'user': 'user', 'guest': 'guest', 'viewer': 'guest',
    'moderator': 'moderator', 'owner': 'owner',
    'staff': 'staff', 'employee': 'staff',
}

FEATURE_KEYWORDS = {
    'login': 'Authentication', 'register': 'Registration', 'signup': 'Registration',
    'dashboard': 'Dashboard', 'analytics': 'Analytics', 'payment': 'Payments',
    'billing': 'Billing', 'search': 'Search', 'filter': 'Filtering',
    'export': 'Export', 'import': 'Import', 'chat': 'Chat', 'messaging': 'Messaging',
    'notification': 'Notifications', 'email': 'Email Notifications', 'sms': 'SMS Notifications',
    'comment': 'Comments', 'like': 'Likes', 'follow': 'Follow', 'post': 'Posts',
    'review': 'Reviews', 'rating': 'Ratings', 'cart': 'Shopping Cart',
    'checkout': 'Checkout', 'order': 'Orders', 'refund': 'Refunds',
    'booking': 'Booking System', 'appointment': 'Appointments', 'reservation': 'Reservations',
    'subscription': 'Subscription', 'multi-tenant': 'Multi-tenancy',
    'map': 'Map View', 'dark mode': 'Dark Mode', 'darkmode': 'Dark Mode',
    'insulated bag': 'Insulated Bags', '5-star': 'Rating System', '5 star': 'Rating System',
    'closest': 'Closest Driver Matching', 'auto-assign': 'Auto Assignment',
    'reject': 'Driver Rejection', 'accept': 'Accept Flow',
    '30 second': '30s Acceptance Timer', '30sec': '30s Acceptance Timer', '30s': '30s Acceptance Timer',
    'magic link': 'Magic Link Auth', 'email auth': 'Email Auth',
    'corporate': 'Corporate 2FA', 'token': '2FA Token',
    '2fa': '2FA Authentication', 'mfa': '2FA Authentication',
    'dark mode': 'Dark Mode Toggle', 'darkmode': 'Dark Mode Toggle',
    'hybrid': 'Hybrid App',
}


class IntentExtractor:
    def extract(self, prompt: str) -> Dict[str, Any]:
        """
        Compiler Frontend: Converts unformatted natural language requirements 
        into a semantic, high-fidelity Intermediate Representation (IR).
        """
        try:
            client = _get_nvidia_client()
            
            system_instruction = (
                "You are the structural Intent Extractor frontend of an enterprise software compiler.\n"
                "Analyze the user requirements and output an un-collapsed Intermediate Representation (IR).\n\n"
                "CRITICAL COMPILER ENFORCEMENT RULES:\n"
                "1. DO NOT simplify roles. If a user asks for 'GlobalAdmin' and 'ClinicManager', do not merge them into a generic 'user' or 'admin'.\n"
                "2. Extract custom database fields, tracking keys, structural hierarchies (e.g. self-referencing relationship joins), and complex conditional validation rules entirely.\n"
                "3. Explicitly list any discovered third-party integrations under 'integrations'.\n"
                "4. Populate the 'ambiguities' or 'assumptions' tracking blocks if critical configuration details are missing.\n"
                "5. If the user intent is vague or incomplete, still produce a minimal but complete IR:\n"
                "   - Include at least one entity (e.g., 'Item').\n"
                "   - Include at least one role (e.g., 'user' with 'read' permission).\n"
                "   - Include a plausible app_type based on keywords.\n"
                "   - Never omit any of the required top-level fields.\n\n"
                "Output ONLY a raw, minified JSON object matching this exact schema structure without markdown wrappers or explanation block strings:\n"
                "{\n"
                "  \"app_name\": \"String\",\n"
                "  \"app_type\": \"String\",\n"
                "  \"features\": [\"List of unique application capabilities requiring functional layers\"],\n"
                "  \"entities\": [\"List of all domain database tables/entities parsed\"],\n"
                "  \"roles\": [{\"name\": \"role_name\", \"permissions\": [\"create\", \"read\", \"update\", \"delete\"]}],\n"
                "  \"integrations\": [\"Third-party external APIs identified\"],\n"
                "  \"ambiguities\": [],\n"
                "  \"assumptions\": []\n"
                "}"
            )
            
            response = client.chat.completions.create(
                model=MODELS["generation"],
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"User Prompt Requirements: {prompt}"}
                ],
                temperature=0.0,  # Ensure absolute compiler determinism
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content
            
            # Programmatically clean deepseek reasoning thought tags if leaked through JSON mode
            if "</thought>" in raw_content:
                raw_content = raw_content.split("</thought>")[-1].strip()
                
            intent_data = json.loads(raw_content)
            return self._validate_and_heal_ir(intent_data)

        except Exception as e:
            logger.error(f"LLM Intent Extraction Compiler Fault, dropping back to rule-based parser: {e}")
            return self.extract_rule_based(prompt)

    def _validate_and_heal_ir(self, ir: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures structural compiler types are consistent before passing to design engine."""
        required_keys = ["features", "entities", "roles", "integrations", "ambiguities", "assumptions"]
        for key in required_keys:
            if key not in ir or not isinstance(ir[key], list):
                ir[key] = []
        
        # Add default content if empty (ensures minimal completeness)
        if not ir["entities"]:
            ir["entities"] = ["Item"]
            ir["assumptions"].append("Added default entity 'Item' because none were extracted")
        if not ir["roles"]:
            ir["roles"] = [{"name": "user", "permissions": ["read"]}]
            ir["assumptions"].append("Added default 'user' role because none were extracted")
        if not ir["features"]:
            ir["features"] = ["Basic CRUD Operations"]
            ir["assumptions"].append("Added basic CRUD feature because none were extracted")
        
        if "app_name" not in ir or not ir["app_name"] or ir["app_name"] in ["MyApp", "App"]:
            ir["app_name"] = "GeneratedEnterpriseApp"
            
        if "app_type" not in ir or not ir["app_type"]:
            ir["app_type"] = "saas"
            
        return ir

    def extract_with_review(self, prompt: str) -> Dict[str, Any]:
        """Fallback wrapper loop supporting downstream minimax review validations."""
        draft = self.extract(prompt)
        try:
            from pipeline.llm import review_with_model
            draft_json = json.dumps(draft)
            corrected, was_fixed = review_with_model(
                draft_json,
                "Ensure app_name, app_type, features[], entities[], roles[], integrations[] are all present and comprehensive"
            )
            if was_fixed:
                draft = json.loads(corrected)
                logger.info(f"Intent Refinement Complete: MiniMax fixed={was_fixed}")
        except Exception as e:
            logger.warning(f"Intent verification check failed: {e}")
        
        return draft
    
    def extract_rule_based(self, prompt: str) -> Dict:
        """Deterministic keyword parser preserved strictly for compiler fallback safety."""
        prompt_lower = prompt.lower()
        entities = self._extract_entities(prompt_lower)
        roles = self._extract_roles(prompt_lower)
        features = self._extract_features(prompt, entities, roles)
        integrations = self._detect_integrations(prompt_lower)
        app_type = self._detect_app_type(prompt_lower)
        
        deduped_features = list(set(features))
        intent = {
            "app_name": self._generate_app_name(prompt),
            "app_type": app_type,
            "features": deduped_features,
            "entities": list(set(entities)),
            "roles": roles,
            "integrations": list(set(integrations)),
            "ambiguities": [],
            "assumptions": []
        }
        
        role_names = [r["name"] for r in roles]
        if len(roles) == 1:
            if "admin" not in role_names:
                intent["assumptions"].append("Added admin role for system management")
                intent["roles"].append({"name": "admin", "permissions": ["create", "read", "update", "delete", "admin"]})
        
        if "magic link" in prompt_lower or "passwordless" in prompt_lower:
            intent["features"] = [f for f in intent["features"] if "magic" not in f.lower() or f == "Magic Link Auth"]
            if "Magic Link Auth" not in intent["features"]:
                intent["features"].append("Magic Link Auth")
        
        if "2fa" in prompt_lower or "corporate" in prompt_lower or "mfa" in prompt_lower:
            if "2FA Authentication" not in intent["features"] and "2FA Token" not in intent["features"]:
                intent["features"].append("2FA Authentication")
        
        if "map" in prompt_lower:
            if "Map View" not in intent["features"]:
                intent["features"].append("Map View")
        
        if "dark mode" in prompt_lower or "darkmode" in prompt_lower:
            if "Dark Mode" not in intent["features"]:
                intent["features"].append("Dark Mode")
        
        return intent
    
    def _extract_entities(self, text: str) -> List[str]:
        found = set()
        entities = []
        
        for keyword, entity_name in ENTITY_KEYWORDS.items():
            if keyword in text and entity_name not in found:
                entities.append(entity_name)
                found.add(entity_name)
        
        if not entities:
            entities.append("items")
        
        return entities
    
    def _extract_roles(self, text: str) -> List[Dict]:
        roles = []
        seen = set()
        
        for keyword, role in ROLE_KEYWORDS.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text) and role not in seen:
                perms = self._get_role_permissions(role)
                roles.append({"name": role, "permissions": perms})
                seen.add(role)
        
        if 'admin' in text and 'user' not in seen and 'customer' not in seen:
            roles.append({"name": "user", "permissions": ["create", "read", "update"]})
        
        if not roles:
            roles.append({"name": "user", "permissions": ["create", "read", "update"]})
        
        return roles
    
    def _get_role_permissions(self, role: str) -> List[str]:
        if role in ('admin', 'global_admin', 'super_admin'):
            return ["create", "read", "update", "delete", "admin"]
        elif role in ('manager', 'clinic_manager'):
            return ["create", "read", "update"]
        elif role in ('guest', 'viewer'):
            return ["read"]
        elif role == 'moderator':
            return ["create", "read", "update", "delete"]
        elif role == 'doctor':
            return ["create", "read", "update"]
        elif role == 'patient':
            return ["create", "read", "update"]
        elif role == 'driver':
            return ["create", "read", "update"]
        elif role == 'staff':
            return ["create", "read", "update"]
        else:
            return ["create", "read", "update"]
    
    def _extract_features(self, prompt: str, entities: List, roles: List) -> List[str]:
        features = []
        prompt_lower = prompt.lower()
        
        for keyword, feature in FEATURE_KEYWORDS.items():
            if keyword.lower() in prompt_lower and feature not in features:
                features.append(feature)
        
        for entity in entities:
            if entity not in ['items', 'auth', 'tokens', 'sessions', 'emails', 'sms', 'webhooks', 'maps', 'locations']:
                features.append(f"{entity.title()} CRUD")
        
        auth_features = set()
        has_passwordless = 'magic link' in prompt_lower or 'passwordless' in prompt_lower
        has_2fa = '2fa' in prompt_lower or 'mfa' in prompt_lower or 'corporate' in prompt_lower or 'token' in prompt_lower
        
        if has_passwordless and has_2fa:
            auth_features.add("Passwordless Magic Link Auth")
            auth_features.add("2FA Authentication")
        elif has_passwordless:
            auth_features.add("Passwordless Magic Link Auth")
        elif has_2fa:
            auth_features.add("2FA Authentication")
        
        if 'driver' in prompt_lower:
            auth_features.add("Driver Status (Online/Offline)")
            if any(k in prompt_lower for k in ['30 second', '30sec', '30s']):
                auth_features.add("30s Driver Acceptance Timer")
            if 'insulated' in prompt_lower or 'bag' in prompt_lower:
                auth_features.add("Insulated Bag Indicator")
        
        if 'reject' in prompt_lower or 'accept' in prompt_lower:
            auth_features.add("Driver Accept/Reject Flow")
        
        if 'closest' in prompt_lower:
            auth_features.add("Closest Driver Matching")
        
        if 'dark mode' in prompt_lower or 'darkmode' in prompt_lower:
            auth_features.add("Dark Mode Toggle")
        
        features.extend(auth_features)
        return list(set(features))
    
    def _detect_integrations(self, text: str) -> List[str]:
        integration_map = {
            'stripe': 'Stripe', 'payment': 'Stripe',
            'email': 'SendGrid', 'mail': 'SendGrid',
            'sms': 'Twilio',
            'auth': 'Auth0', 'auth0': 'Auth0',
            'analytics': 'Mixpanel',
            'slack': 'Slack',
            'github': 'GitHub',
            'google': 'Google OAuth',
            'facebook': 'Facebook', 'instagram': 'Instagram',
            'mapbox': 'Mapbox', 'maps': 'Mapbox', 'map': 'Mapbox',
            'twilio': 'Twilio', 'sendgrid': 'SendGrid',
            'razorpay': 'Razorpay',
        }
        return [name for key, name in integration_map.items() if key in text]
    
    def _detect_app_type(self, text: str) -> str:
        type_signatures = {
            'crm': ['crm', 'customer relationship', 'contacts', 'leads', 'deals'],
            'ecommerce': ['ecommerce', 'shop', 'store', 'cart', 'checkout'],
            'saas': ['saas', 'subscription', 'multi-tenant'],
            'dashboard': ['dashboard', 'analytics', 'metrics', 'reporting'],
            'marketplace': ['marketplace', 'vendor', 'seller'],
            'social': ['social', 'post', 'like', 'comment', 'follow', 'feed'],
            'healthcare': ['patient', 'doctor', 'medical', 'health', 'clinic'],
            'booking': ['booking', 'appointment', 'reservation'],
            'ride': ['ride', 'driver', 'taxi', 'cab'],
            'delivery': ['delivery', 'food delivery', 'courier'],
        }
        for app_type, signatures in type_signatures.items():
            if any(sig in text for sig in signatures):
                return app_type
        return 'saas'
    
    def _generate_app_name(self, prompt: str) -> str:
        stop_words = {'crm', 'user', 'admin', 'build', 'create', 'make', 'with', 'that', 'this', 'the', 'a', 'an', 'for', 'and', 'or', 'but', 'app', 'application'}
        words = [w for w in prompt.split() if len(w) > 3 and w.lower() not in stop_words]
        if len(words) >= 3:
            return ''.join(w.capitalize() for w in words[:3])
        elif words:
            return ''.join(w.capitalize() for w in words) + 'App'
        return 'MyApp'
