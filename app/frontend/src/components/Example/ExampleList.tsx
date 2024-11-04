import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = [
    "What is required for a payment company to become a PayFac?",
    "Write an email to a customer of Flatirons Bank who has failed to make their installment loan payment.",
    "We have an upcoming CRA audit from the OCC, what should the management team do to prepare?",
    "Create a 5-question test on the Bank Secrecy Act, provide 3 possible answers for each question."
];

const GPT4V_EXAMPLES: string[] = [
    "What is required for a payment company to become a PayFac?",
    "Write an email to a customer of Flatirons Bank who has failed to make their installment loan payment.",
    "We have an upcoming CRA audit from the OCC, what should the management team do to prepare?",
    "Create a 5-question test on the Bank Secrecy Act, provide 3 possible answers for each question."
];

interface Props {
    onExampleClicked: (value: string) => void;
    useGPT4V?: boolean;
}

export const ExampleList = ({ onExampleClicked, useGPT4V }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {(useGPT4V ? GPT4V_EXAMPLES : DEFAULT_EXAMPLES).map((question, i) => (
                <li key={i}>
                    <Example text={question} value={question} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
