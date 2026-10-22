import Foundation
import Security

/// Stores the Hugging Face token in the user's login keychain.
enum Keychain {
    private static let service = "local.imagestudio"
    private static let account = "huggingface-token"

    private static var baseQuery: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: account]
    }

    static func read() -> String? {
        var query = baseQuery
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func save(_ value: String) throws {
        let attributes = [kSecValueData as String: Data(value.utf8)]
        var status = SecItemUpdate(baseQuery as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            var query = baseQuery
            query[kSecValueData as String] = Data(value.utf8)
            status = SecItemAdd(query as CFDictionary, nil)
        }
        try check(status)
    }

    static func delete() throws {
        let status = SecItemDelete(baseQuery as CFDictionary)
        if status != errSecItemNotFound { try check(status) }
    }

    private static func check(_ status: OSStatus) throws {
        guard status != errSecSuccess else { return }
        let message = SecCopyErrorMessageString(status, nil) as String? ?? "Keychain error \(status)"
        throw BackendError(message: message)
    }
}
