import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = [
    "Do you have the bank secrecy act regulation?",
    "Please write an email to a customer who has made a payment with an account with insufficient funds?",
    "Create a detailed policy for outbound debt collection agents.",
    "What KYC regulations are required for payment companies?"
];

const GPT4V_EXAMPLES: string[] = [
    "Do you have the bank secrecy act regulation?",
    "Please write an email to a customer who has made a payment with an account with insufficient funds?",
    "Create a detailed policy for outbound debt collection agents.",
    "What KYC regulations are required for payment companies?"
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
