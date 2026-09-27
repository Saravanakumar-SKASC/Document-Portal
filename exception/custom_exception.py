import sys
import traceback


class DocumentPortalException(Exception):
    """
    Custom exception for Document Portal.

    Records the file and line where the underlying error happened plus the full
    traceback. Safe to raise both inside an ``except`` block (it captures the
    active exception) and outside one (for validation errors with no cause).

        raise DocumentPortalException("Error reading PDF", e) from e
        raise DocumentPortalException("Invalid file type. Only PDFs are allowed.")
    """

    def __init__(self, error_message, error_details: BaseException | None = None):
        super().__init__(str(error_message))
        self.error_message = str(error_message)

        exc_type, exc_value, exc_tb = sys.exc_info()
        if error_details is not None and isinstance(error_details, BaseException):
            exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__

        if exc_tb is not None:
            last = exc_tb
            while last.tb_next is not None:  # walk to the frame where the error was raised
                last = last.tb_next
            self.file_name = last.tb_frame.f_code.co_filename
            self.lineno = last.tb_lineno
            self.traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            frame = sys._getframe(1)
            self.file_name = frame.f_code.co_filename
            self.lineno = frame.f_lineno
            self.traceback_str = ""

    def __str__(self):
        message = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            message += f"\nTraceback:\n{self.traceback_str}"
        return message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        print(DocumentPortalException("Division failed", e))
