#!/usr/bin/env python3
"""Reject configuration controls outside the one-phase free-surface scope."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
import json
import re
import sys
from typing import Any
import xml.etree.ElementTree as ET


UNSUPPORTED_SCOPE_DIAGNOSTIC = "unsupported_two_phase_or_jump_free_surface_scope"
UNSUPPORTED_CONTROL_MARKERS = (
    "twophase",
    "twofluid",
    "multiphase",
    "pressureenrichment",
    "jump",
    "gas",
    "gasdensity",
    "gasviscosity",
)
UNSUPPORTED_SCOPE_VALUE_MARKERS = UNSUPPORTED_CONTROL_MARKERS
SCOPE_CONTROL_NAMES = {
    "capability",
    "capabilityscope",
    "fluidmodel",
    "formulation",
    "freesurfacemodel",
    "freesurfacephysicalmodel",
    "implementation",
    "interfacephysics",
    "interfacemodel",
    "materialmodel",
    "model",
    "modelscope",
    "phasemodel",
    "physicalmodel",
    "physics",
    "physicsscope",
    "scope",
    "type",
}
WRAPPER_CONTROL_NAMES = {
    "key",
    "name",
    "option",
    "parameter",
}
WRAPPER_VALUE_NAMES = {
    "setting",
    "value",
    "values",
}


class UnsupportedFreeSurfaceScope(ValueError):
    """Raised when a configuration requests physics outside one-phase scope."""


def normalized_token(value: str) -> str:
    """Return a case- and punctuation-insensitive configuration token."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def local_xml_name(value: str) -> str:
    """Strip an optional XML namespace from a tag or attribute name."""
    return value.rsplit("}", 1)[-1]


def reject_unsupported_scope() -> None:
    """Raise the stable unsupported-scope diagnostic."""
    raise UnsupportedFreeSurfaceScope(UNSUPPORTED_SCOPE_DIAGNOSTIC)


def validate_control_name(name: str) -> str:
    """Reject an unsupported configuration key or XML tag."""
    token = normalized_token(name)
    if any(marker in token for marker in UNSUPPORTED_CONTROL_MARKERS):
        reject_unsupported_scope()
    return token


def validate_scope_value(value: Any) -> None:
    """Reject an unsupported value assigned to a model/scope control."""
    for scalar in scalar_strings(value):
        token = normalized_token(scalar)
        if any(marker in token for marker in UNSUPPORTED_SCOPE_VALUE_MARKERS):
            reject_unsupported_scope()


def validate_wrapper_value(value: Any) -> None:
    """Reject an unsupported control name stored in a wrapper value."""
    for scalar in scalar_strings(value):
        validate_control_name(scalar)


def scalar_strings(value: Any) -> Iterator[str]:
    """Yield scalar strings from nested mapping and sequence values."""
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from scalar_strings(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from scalar_strings(item)


def wrapped_scope_controls(value: Any) -> set[str]:
    """Return known scope-control tokens from a structured wrapper selector."""
    controls: set[str] = set()
    for scalar in scalar_strings(value):
        token = validate_control_name(scalar)
        if token in SCOPE_CONTROL_NAMES:
            controls.add(token)
    return controls


def validate_mapping_wrapper_pairs(value: Mapping[str, Any]) -> None:
    """Couple name/key/option/parameter fields to sibling value fields."""
    normalized_items = [
        (validate_control_name(name), item)
        for name, item in value.items()
        if isinstance(name, str)
    ]
    wrapped_scopes = {
        scope
        for token, item in normalized_items
        if token in WRAPPER_CONTROL_NAMES
        for scope in wrapped_scope_controls(item)
    }
    if not wrapped_scopes:
        return
    for token, item in normalized_items:
        if token in WRAPPER_VALUE_NAMES:
            validate_scope_value(item)


def validate_value(value: Any) -> None:
    """Recursively validate a parsed JSON value or configuration mapping."""
    if isinstance(value, Mapping):
        validate_mapping_wrapper_pairs(value)
        for name, item in value.items():
            if not isinstance(name, str):
                raise ValueError("configuration mapping keys must be strings")
            token = validate_control_name(name)
            if token in SCOPE_CONTROL_NAMES:
                validate_scope_value(item)
            if token in WRAPPER_CONTROL_NAMES:
                validate_wrapper_value(item)
            validate_value(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            validate_value(item)


def validate_config_mapping(value: Mapping[str, Any]) -> None:
    """Validate an already parsed configuration mapping."""
    if not isinstance(value, Mapping):
        raise ValueError("configuration payload must be a mapping")
    validate_value(value)


def reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Reject duplicate JSON keys before scope validation."""
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def validate_json_config(payload: str) -> None:
    """Parse and validate a JSON configuration."""
    value = json.loads(
        payload,
        object_pairs_hook=reject_duplicate_json_keys,
    )
    validate_config_mapping(value)


def xml_scalar_strings(element: ET.Element) -> Iterator[str]:
    """Yield text, attribute values, descendants, and child tails."""
    if element.text is not None:
        yield element.text
    for attribute_value in element.attrib.values():
        yield attribute_value
    for child in element:
        yield from xml_scalar_strings(child)
        if child.tail is not None:
            yield child.tail


def validate_xml_scope_subtree(element: ET.Element) -> None:
    """Validate every scalar value nested below a known scope control."""
    validate_scope_value(list(xml_scalar_strings(element)))


def validate_xml_config(payload: str) -> None:
    """Parse and validate an XML configuration."""
    root = ET.fromstring(payload)
    for element in root.iter():
        element_name = local_xml_name(element.tag)
        element_token = validate_control_name(element_name)
        normalized_attributes = [
            (
                validate_control_name(local_xml_name(attribute_name)),
                attribute_value,
            )
            for attribute_name, attribute_value in element.attrib.items()
        ]
        normalized_children = [
            (validate_control_name(local_xml_name(child.tag)), child)
            for child in element
        ]
        wrapped_scopes = {
            scope
            for attribute_token, attribute_value in normalized_attributes
            if attribute_token in WRAPPER_CONTROL_NAMES
            for scope in wrapped_scope_controls(attribute_value)
        }
        wrapped_scopes.update(
            scope
            for child_token, child in normalized_children
            if child_token in WRAPPER_CONTROL_NAMES
            for scope in wrapped_scope_controls(list(xml_scalar_strings(child)))
        )
        if wrapped_scopes:
            validate_xml_scope_subtree(element)
        if element_token in SCOPE_CONTROL_NAMES:
            validate_xml_scope_subtree(element)
        if element_token in WRAPPER_CONTROL_NAMES and element.text is not None:
            validate_wrapper_value(element.text)
        for attribute_token, attribute_value in normalized_attributes:
            if attribute_token in SCOPE_CONTROL_NAMES:
                validate_scope_value(attribute_value)
            if attribute_token in WRAPPER_CONTROL_NAMES:
                validate_wrapper_value(attribute_value)


def validate_payload(payload_format: str, payload: Any) -> None:
    """Parse and validate one supported representative payload format."""
    if payload_format == "xml":
        if not isinstance(payload, str):
            raise ValueError("XML payload must be a string")
        validate_xml_config(payload)
        return
    if payload_format == "json":
        if not isinstance(payload, str):
            raise ValueError("JSON payload must be a string")
        validate_json_config(payload)
        return
    if payload_format == "mapping":
        validate_config_mapping(payload)
        return
    raise ValueError(f"unsupported payload format: {payload_format}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("xml", "json", "mapping"),
        required=True,
        dest="payload_format",
    )
    parser.add_argument(
        "payload",
        nargs="?",
        help="configuration text; stdin is used when omitted",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    payload = arguments.payload if arguments.payload is not None else sys.stdin.read()
    if arguments.payload_format == "mapping":
        payload = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_json_keys,
        )
    try:
        validate_payload(arguments.payload_format, payload)
    except UnsupportedFreeSurfaceScope as error:
        print(str(error), file=sys.stderr)
        return 2
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    print("one_phase_scope_guard_pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
