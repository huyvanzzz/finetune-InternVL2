from training_runtime import enable_gradient_checkpointing


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)


class WrapperWithoutSupport:
    def gradient_checkpointing_enable(self):
        raise ValueError("WrapperModel does not support gradient checkpointing.")


class LanguageModelWithSupport:
    def __init__(self):
        self.enabled = False

    def gradient_checkpointing_enable(self):
        self.enabled = True


class LanguageModelWithoutMethod:
    pass


class DummyModel:
    def __init__(self, language_model):
        self.language_model = language_model


def test_enable_gradient_checkpointing_falls_back_to_language_model():
    logger = DummyLogger()
    language_model = LanguageModelWithSupport()
    model = DummyModel(language_model=language_model)
    model.gradient_checkpointing_enable = WrapperWithoutSupport().gradient_checkpointing_enable

    enabled_on = enable_gradient_checkpointing(model, enabled=True, logger=logger)

    assert enabled_on == "language_model"
    assert language_model.enabled is True
    assert any("language_model" in msg for msg in logger.infos)


def test_enable_gradient_checkpointing_warns_when_unavailable():
    logger = DummyLogger()
    model = DummyModel(language_model=LanguageModelWithoutMethod())

    enabled_on = enable_gradient_checkpointing(model, enabled=True, logger=logger)

    assert enabled_on is None
    assert len(logger.warnings) == 1


def test_enable_gradient_checkpointing_noop_when_disabled():
    logger = DummyLogger()
    language_model = LanguageModelWithSupport()
    model = DummyModel(language_model=language_model)

    enabled_on = enable_gradient_checkpointing(model, enabled=False, logger=logger)

    assert enabled_on is None
    assert language_model.enabled is False
    assert logger.infos == []
    assert logger.warnings == []
