"""Input validation schemas for API requests."""

from marshmallow import Schema, fields, validate


class TrainingRequestSchema(Schema):
    """Schema for POST /api/train request body."""

    episodes = fields.Int(
        load_default=100,
        validate=validate.Range(min=1, max=10000),
    )
    nodes = fields.Int(
        load_default=550,
        validate=validate.Range(min=10, max=10000),
    )
    learning_rate = fields.Float(
        load_default=1e-4,
        validate=validate.Range(min=1e-6, max=1e-1),
    )
    gamma = fields.Float(
        load_default=0.99,
        validate=validate.Range(min=0.0, max=1.0),
    )
    batch_size = fields.Int(
        load_default=64,
        validate=validate.Range(min=8, max=512),
    )
    death_threshold = fields.Float(
        load_default=0.3,
        validate=validate.Range(min=0.0, max=1.0),
    )
    max_steps = fields.Int(
        load_default=1000,
        validate=validate.Range(min=50, max=10000),
    )
    seed = fields.Int(load_default=42)
    model_type = fields.Str(
        load_default="ddqn",
        validate=validate.OneOf(["dqn", "ddqn"]),
    )
