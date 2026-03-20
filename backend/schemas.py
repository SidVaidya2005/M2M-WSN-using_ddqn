"""Input validation schemas for API endpoints."""

from marshmallow import Schema, fields, validate


class TrainingRequestSchema(Schema):
    """Schema for training request validation."""
    
    episodes = fields.Int(
        required=False,
        validate=validate.Range(min=1, max=10000),
        missing=100,
        load_default=100,
    )
    nodes = fields.Int(
        required=False,
        validate=validate.Range(min=10, max=10000),
        missing=550,
        load_default=550,
    )
    learning_rate = fields.Float(
        required=False,
        validate=validate.Range(min=1e-6, max=1e-1),
        missing=1e-4,
        load_default=1e-4,
    )
    gamma = fields.Float(
        required=False,
        validate=validate.Range(min=0.0, max=1.0),
        missing=0.99,
        load_default=0.99,
    )
    batch_size = fields.Int(
        required=False,
        validate=validate.Range(min=8, max=512),
        missing=64,
        load_default=64,
    )
    seed = fields.Int(
        required=False,
        missing=42,
        load_default=42,
    )


class ConfigUpdateSchema(Schema):
    """Schema for configuration update requests."""
    
    episodes = fields.Int(validate=validate.Range(min=1, max=10000), allow_none=True)
    learning_rate = fields.Float(validate=validate.Range(min=1e-6, max=1e-1), allow_none=True)
    gamma = fields.Float(validate=validate.Range(min=0.0, max=1.0), allow_none=True)
    batch_size = fields.Int(validate=validate.Range(min=8, max=512), allow_none=True)
